"""Training a blend: every stage in memory, from input files to the fitted model."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .core import climatology as climatology_stage, features, onsets, parallel
from .core.climatology import Climatology
from .config import BlendConfig
from .core.features import DailyInputs, Weekly
from .core.onsets import ForecastOnsets, ObservedOnsets
from .core.reference import Reference

log = logging.getLogger(__name__)


@dataclass
class Resources:
    """Processes the stages may use, and the memory their parallel work shares."""

    workers: int
    budget_bytes: int | None

    @classmethod
    def default(cls) -> "Resources":
        return cls(*parallel.default_resources())


@dataclass
class Inputs:
    """Everything the blend is fitted on, stage by stage."""

    reference: Reference
    observed: ObservedOnsets
    forecasts: dict[str, ForecastOnsets]
    climatology: Climatology
    weekly: Weekly
    combined: pd.DataFrame | None = None   # the day-level table, only when asked for
    timings: dict[str, float] = field(default_factory=dict)
    forest_inputs: DailyInputs | None = None   # the random forest blend's, when it is configured


STAGE_TABLES = ("observed", "forecasts", "climatology", "weekly")


def prepare(
    config: BlendConfig,
    *,
    resources: Resources | None = None,
    test_years: tuple[int, int] | None = None,
    keep: tuple[str, ...] = STAGE_TABLES,
    daily: bool = False,
) -> Inputs:
    """Observed and forecast onsets, the climatology and the weekly
    features for a config; with daily, also the day-level table.

    Forecast and climatology arrays not in `keep` are released once the
    features are built; the observed onsets and weekly features are always
    kept. test_years overrides the climatology's test window (a live season)."""
    resources = resources or Resources.default()
    workers, budget = resources.workers, resources.budget_bytes
    timings: dict[str, float] = {}

    def lap(stage: str, started: float) -> None:
        timings[stage] = time.perf_counter() - started
        log.info("%s in %.1fs", stage, timings[stage])

    started = time.perf_counter()
    reference = Reference.resolve(config)
    observed = onsets.observed(config, reference=reference, workers=workers, budget_bytes=budget)
    lap("observed onsets", started)

    # Under the inner join only units the climatology keeps survive, so the
    # forecasts are read for those alone.
    blendable = observed.blendable(config.climatology.min_onset_years) if config.features.join == "inner" else None
    forecasts = {}
    for model in config.models:
        started = time.perf_counter()
        forecasts[model.name] = onsets.forecast(
            config, model, keep_ids=blendable, daily=daily, reference=reference,
            workers=workers, budget_bytes=budget,
        )
        lap(f"{model.name} forecast onsets", started)

    started = time.perf_counter()
    forecast_years = sorted({y for f in forecasts.values() for y in f.years})
    climatology = climatology_stage.build(
        config, observed, forecast_years, test_years=test_years, workers=workers
    )
    lap("climatology", started)

    started = time.perf_counter()
    weekly = features.weekly(config, climatology, observed, forecasts)
    lap("weekly features", started)
    forest_inputs = None
    if config.evaluation.random_forest is not None:
        started = time.perf_counter()
        forest_inputs = features.daily_inputs(
            config, climatology, observed, forecasts, max_day=config.evaluation.random_forest.max_day
        )
        lap("random forest inputs", started)
    combined = None
    if daily:
        started = time.perf_counter()
        combined = features.daily(config, climatology, observed, forecasts)
        lap("day-level table", started)
    if "forecasts" not in keep:
        for model_onsets in forecasts.values():
            model_onsets.release()
    if "climatology" not in keep:
        climatology.release()
    return Inputs(
        reference=reference,
        observed=observed,
        forecasts=forecasts,
        climatology=climatology,
        weekly=weekly,
        combined=combined,
        timings=timings,
        forest_inputs=forest_inputs,
    )


def dissemination_table(config: BlendConfig, weekly: Weekly) -> pd.DataFrame:
    """The units the blend trains on and is scored on: the configured table, or
    every unit of the weekly features."""
    if config.domain.dissemination is None:
        return pd.DataFrame({"adm3_name": sorted({str(unit) for unit in weekly.units})})
    return pd.read_csv(config.path(config.domain.dissemination.file), dtype=str)


@dataclass
class TrainingResult:
    config: BlendConfig
    inputs: Inputs
    evaluation: object      # evaluation.Evaluation
    blend: object           # blend.TrainedBlend

    def save(self, directory, *, intermediates=("weekly",)) -> Path:
        """Write the evaluation, the trained blend (trained.blend) and the chosen
        intermediates into a new directory, all or nothing: the weekly
        features (as the legacy table) and day-level ("combined", when built)
        tables as pickles, the observed onsets as NetCDF."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            self.evaluation.save(staging)
            self.blend.save(staging / TRAINED_BLEND_FILENAME)
            self.blend.config.to_yaml(staging / "config.yml")
            for name in intermediates:
                if name == "observed":
                    self.inputs.observed.data.to_netcdf(staging / "observed_onsets.nc")
                elif name == "combined":
                    if self.inputs.combined is None:
                        raise ValueError("the day-level table was not built (train(..., daily=True))")
                    self.inputs.combined.to_pickle(staging / "combined_wide.pkl")
                elif name == "weekly":
                    self.inputs.weekly.frame().to_pickle(staging / "weekly_training_input.pkl")
                else:
                    raise ValueError(f"unknown intermediate {name!r}")
            if directory.exists():
                directory.rmdir()
            os.replace(staging, directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return directory


TRAINED_BLEND_FILENAME = "trained.blend"


def train(config: BlendConfig, *, resources: Resources | None = None, daily: bool = False) -> TrainingResult:
    """Every stage of a blend, in memory: inputs, cross-validated evaluation,
    and the operational model fitted on every training year. daily also
    builds the day-level table (for export or day-level models)."""
    from .core import _specs
    from .trained import TrainedBlend
    from .blend import random_forest
    from .evaluation import evaluate

    resources = resources or Resources.default()
    inputs = prepare(config, resources=resources, keep=("observed", "weekly"), daily=daily)
    cv_spec = _specs.cv_spec(config)
    dissemination = dissemination_table(config, inputs.weekly)

    # The final fit runs alongside the global CV folds, on their design matrix.
    started = time.perf_counter()
    forest = config.evaluation.random_forest
    evaluation = evaluate(
        inputs.weekly, cv_spec, dissemination, workers=resources.workers,
        budget_bytes=_budget_now(resources), final_model=config.final_formula(),
        forest=None if forest is None else random_forest.Params.from_config(forest, config.features.bins.days),
        forest_inputs=inputs.forest_inputs,
    )
    inputs.timings["evaluation and final fit"] = time.perf_counter() - started
    log.info("evaluation and final fit in %.1fs", inputs.timings["evaluation and final fit"])
    blend = TrainedBlend.from_training(config, inputs, evaluation.final_model, forest=evaluation.final_forest)
    return TrainingResult(config=config, inputs=inputs, evaluation=evaluation, blend=blend)


def _budget_now(resources: Resources) -> int | None:
    """The memory budget, within half of what is available now that the
    stage arrays exist."""
    budgets = [b for b in (resources.budget_bytes, parallel.default_resources()[1]) if b is not None]
    return min(budgets) if budgets else None
