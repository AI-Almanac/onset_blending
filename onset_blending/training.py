"""Training a blend: every stage in memory, from input files to the fitted model."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path


from .core import climatology as climatology_stage, features, onsets, parallel
from .core.climatology import Climatology
from .config import BlendConfig
from .core.features import DailyInputs, Weekly
from .core.onsets import ForecastOnsets, ObservedOnsets
from .core.reference import Reference

log = logging.getLogger(__name__)


@dataclass
class Resources:
    """Processes the stages may use, the memory their parallel work shares,
    and where results that outlive a run (computed onset thresholds) are kept."""

    workers: int
    budget_bytes: int | None
    cache_dir: str | Path | None = None

    @classmethod
    def default(cls, *, cache_dir: str | Path | None = None) -> "Resources":
        return cls(*parallel.default_resources(), cache_dir=cache_dir)


@dataclass
class Inputs:
    """Everything the blend is fitted on, stage by stage."""

    reference: Reference
    observed: ObservedOnsets
    forecasts: dict[str, ForecastOnsets]
    climatology: Climatology
    weekly: Weekly
    timings: dict[str, float] = field(default_factory=dict)
    forest_inputs: DailyInputs | None = None   # the random forest blend's, when it is configured


STAGE_TABLES = ("observed", "forecasts", "climatology", "weekly")


def prepare(
    config: BlendConfig,
    *,
    resources: Resources | None = None,
    test_years: tuple[int, int] | None = None,
    keep: tuple[str, ...] = STAGE_TABLES,
) -> Inputs:
    """Observed and forecast onsets, the climatology and the weekly
    features for a config (and the random forest's inputs when it has one).

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
    reference = Reference.resolve(config, workers=workers, cache_dir=resources.cache_dir)
    observed = onsets.observed(config, reference=reference, workers=workers, budget_bytes=budget)
    lap("observed onsets", started)

    # Only units the climatology keeps make samples, so the forecasts are
    # read for those alone.
    blendable = observed.blendable(config.climatology.min_onset_years)
    forecasts = {}
    for model in config.models:
        started = time.perf_counter()
        forecasts[model.name] = onsets.forecast(
            config, model, keep_ids=blendable, reference=reference,
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
        timings=timings,
        forest_inputs=forest_inputs,
    )


@dataclass
class TrainingResult:
    config: BlendConfig
    inputs: Inputs
    evaluation: object      # evaluation.Evaluation
    blend: object           # blend.TrainedBlend

    def save(self, directory, *, intermediates=("weekly",)) -> Path:
        """Write the evaluation, the trained blend (trained.blend) and the chosen
        intermediates into a new directory, all or nothing: the weekly
        features (Weekly.open reads them) and the observed onsets as NetCDF,
        nothing else."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            self.evaluation.save(staging, pickles=self.config.outputs.pickles)
            self.blend.save(staging / TRAINED_BLEND_FILENAME)
            self.blend.config.to_yaml(staging / "config.yml")
            for name in intermediates:
                if name == "observed":
                    self.inputs.observed.data.to_netcdf(staging / "observed_onsets.nc")
                elif name == "weekly":
                    self.inputs.weekly.to_netcdf(staging / "weekly_features.nc")
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


def train(config: BlendConfig, *, resources: Resources | None = None) -> TrainingResult:
    """Every stage of a blend, in memory: inputs, cross-validated evaluation,
    and the operational model (and the random forest, when configured)
    fitted on every training year."""
    from .evaluation import evaluate
    from .trained import TrainedBlend

    resources = resources or Resources.default()
    inputs = prepare(config, resources=resources, keep=("observed", "weekly"))

    # The final fit runs alongside the global CV folds, on their design matrix.
    started = time.perf_counter()
    evaluation = evaluate(
        inputs.weekly, config, workers=resources.workers, budget_bytes=_budget_now(resources), final=True,
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
