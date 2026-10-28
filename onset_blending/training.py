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

from . import _parallel, climatology as climatology_stage, features, onsets
from .climatology import Climatology
from .config import BlendConfig
from .onsets import ForecastOnsets

log = logging.getLogger(__name__)


@dataclass
class Resources:
    """Processes the stages may use, and the memory their parallel work shares."""

    workers: int
    budget_bytes: int | None

    @classmethod
    def default(cls) -> "Resources":
        return cls(*_parallel.default_resources())


@dataclass
class Inputs:
    """Everything the blend is fitted on, stage by stage."""

    observed: pd.DataFrame
    forecasts: dict[str, ForecastOnsets]
    climatology: Climatology
    combined: pd.DataFrame
    weekly: pd.DataFrame
    timings: dict[str, float] = field(default_factory=dict)


STAGE_TABLES = ("observed", "forecasts", "climatology", "combined", "weekly")


def prepare(
    config: BlendConfig,
    *,
    resources: Resources | None = None,
    test_years: tuple[int, int] | None = None,
    keep: tuple[str, ...] = STAGE_TABLES,
) -> Inputs:
    """Observed and forecast onsets, the climatology, and the day-level and
    weekly feature tables for a config.

    Tables not in `keep` are released once the stage reading them has run
    (forecasts and climatology after the day-level table), which bounds
    memory; the observed, combined and weekly tables are always kept.
    test_years overrides the climatology's test window (a live season)."""
    resources = resources or Resources.default()
    workers, budget = resources.workers, resources.budget_bytes
    timings: dict[str, float] = {}

    def lap(stage: str, started: float) -> None:
        timings[stage] = time.perf_counter() - started
        log.info("%s in %.1fs", stage, timings[stage])

    started = time.perf_counter()
    observed = onsets.observed(config, workers=workers, budget_bytes=budget)
    lap("observed onsets", started)

    # Under the inner join only units the climatology keeps survive, so the
    # forecasts are read for those alone.
    keep = (
        onsets.blendable_ids(observed, config.climatology.min_onset_years)
        if config.features.join == "inner"
        else None
    )
    forecasts = {}
    for model in config.models:
        started = time.perf_counter()
        forecasts[model.name] = onsets.forecast(
            config, model, keep_ids=keep, workers=workers, budget_bytes=budget
        )
        lap(f"{model.name} forecast onsets", started)

    started = time.perf_counter()
    forecast_years = sorted({y for f in forecasts.values() for y in f.years})
    climatology = climatology_stage.build(
        config, observed, forecast_years, test_years=test_years, workers=workers
    )
    lap("climatology", started)

    started = time.perf_counter()
    combined = features.combine(
        config, climatology, observed, forecasts, workers=workers, budget_bytes=budget
    )
    lap("day-level features", started)
    if "forecasts" not in keep:
        for model_onsets in forecasts.values():
            model_onsets.release()
    if "climatology" not in keep:
        climatology.release()

    started = time.perf_counter()
    weekly = features.weekly(config, combined)
    lap("weekly features", started)
    return Inputs(
        observed=observed,
        forecasts=forecasts,
        climatology=climatology,
        combined=combined,
        weekly=weekly,
        timings=timings,
    )


def dissemination_table(config: BlendConfig, weekly: pd.DataFrame) -> pd.DataFrame:
    """The units the blend trains on and is scored on: the configured table, or
    every unit of the weekly table."""
    if config.domain.dissemination is None:
        return pd.DataFrame({"adm3_name": sorted(weekly["id"].astype(str).unique())})
    return pd.read_csv(config.path(config.domain.dissemination.file), dtype=str)


@dataclass
class TrainingResult:
    config: BlendConfig
    inputs: Inputs
    evaluation: object      # evaluation.Evaluation
    blend: object           # blend.TrainedBlend

    def save(self, directory, *, intermediates=("combined", "weekly")) -> Path:
        """Write the evaluation, the trained blend (trained.blend) and the chosen
        intermediates into a new directory, all or nothing."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            self.evaluation.save(staging)
            self.blend.save(staging / TRAINED_BLEND_FILENAME)
            self.blend.config.to_yaml(staging / "config.yml")
            tables = {
                "observed": ("observed_onsets.pkl", self.inputs.observed),
                "combined": ("combined_wide.pkl", self.inputs.combined),
                "weekly": ("weekly_training_input.pkl", self.inputs.weekly),
            }
            for name in intermediates:
                filename, table = tables[name]
                table.to_pickle(staging / filename)
            if directory.exists():
                directory.rmdir()
            os.replace(staging, directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return directory


TRAINED_BLEND_FILENAME = "trained.blend"


def train(config: BlendConfig, *, resources: Resources | None = None) -> TrainingResult:
    """Every stage of a blend, in memory: inputs, cross-validated evaluation,
    and the operational model fitted on every training year."""
    from . import _specs
    from .blend import TrainedBlend
    from .evaluation import evaluate
    from .model import fit_final

    resources = resources or Resources.default()
    inputs = prepare(config, resources=resources, keep=("observed", "combined", "weekly"))
    cv_spec = _specs.cv_spec(config)
    dissemination = dissemination_table(config, inputs.weekly)

    started = time.perf_counter()
    evaluation = evaluate(inputs.weekly, cv_spec, dissemination, workers=_cv_workers(inputs.weekly, resources))
    inputs.timings["evaluation"] = time.perf_counter() - started
    log.info("evaluation in %.1fs", inputs.timings["evaluation"])

    started = time.perf_counter()
    fitted = fit_final(inputs.weekly, cv_spec, dissemination, model_name=config.final_formula())
    inputs.timings["final fit"] = time.perf_counter() - started
    blend = TrainedBlend.from_training(config, inputs, fitted)
    return TrainingResult(config=config, inputs=inputs, evaluation=evaluation, blend=blend)


# Copies of the weekly table the evaluation's main process and each CV fold
# worker hold.
_MAIN_TABLE_COPIES = 3
_FOLD_WORKER_TABLE_COPIES = 3


def _cv_workers(weekly: pd.DataFrame, resources: Resources) -> int:
    """CV fold workers whose copies of the weekly table fit the memory budget,
    and the memory available now that the stage tables exist."""
    budgets = [b for b in (resources.budget_bytes, _parallel.default_resources()[1]) if b is not None]
    if not budgets:
        return resources.workers
    budget = min(budgets)
    table = int(weekly.memory_usage(deep=True).sum())
    spare = budget - _MAIN_TABLE_COPIES * table
    return max(1, min(resources.workers, spare // max(_FOLD_WORKER_TABLE_COPIES * table, 1)))
