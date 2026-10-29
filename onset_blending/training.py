"""Training a blend: every stage in memory, from input files to the fitted model."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

from . import _parallel, climatology as climatology_stage, features, onsets
from .climatology import Climatology
from . import config as cfg
from .config import BlendConfig
from .onsets import ForecastOnsets, ObservedOnsets
from .reference import Reference

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

    reference: Reference
    observed: ObservedOnsets
    forecasts: dict[str, ForecastOnsets]
    climatology: Climatology
    weekly: pd.DataFrame
    combined: pd.DataFrame | None = None   # the day-level table, only when asked for
    timings: dict[str, float] = field(default_factory=dict)


STAGE_TABLES = ("observed", "forecasts", "climatology", "weekly")


def prepare(
    config: BlendConfig,
    *,
    resources: Resources | None = None,
    test_years: tuple[int, int] | None = None,
    keep: tuple[str, ...] = STAGE_TABLES,
    daily: bool = False,
) -> Inputs:
    """Observed and forecast onsets, the climatology and the weekly feature
    table for a config; with daily, also the day-level table.

    Forecast and climatology arrays not in `keep` are released once the
    feature tables are built; the observed onsets and weekly table are always
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

    def save(self, directory, *, intermediates=("weekly",)) -> Path:
        """Write the evaluation, the trained blend (trained.blend) and the chosen
        intermediates into a new directory, all or nothing: the weekly and
        day-level ("combined", when built) tables as pickles, the observed
        onsets as NetCDF."""
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
                    self.inputs.weekly.to_pickle(staging / "weekly_training_input.pkl")
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
    from . import _specs
    from .blend import TrainedBlend
    from .evaluation import evaluate
    from .model import fit_final

    resources = resources or Resources.default()
    inputs = prepare(config, resources=resources, keep=("observed", "weekly"), daily=daily)
    cv_spec = _specs.cv_spec(config)
    dissemination = dissemination_table(config, inputs.weekly)

    # The final fit needs only the training rows' formula columns, so it runs
    # in its own process alongside the evaluation (as the separate
    # 3_fit_final_model.py did beside 1_blend_evaluation.py).
    started = time.perf_counter()
    # One BLAS thread: the fit's coefficients do not depend on the thread
    # count (checked on India), and it then takes no core from the evaluation.
    with ProcessPoolExecutor(
        max_workers=1, mp_context=get_context("spawn"), initializer=_parallel.limit_blas_threads, initargs=(1,)
    ) as pool:
        final = pool.submit(
            fit_final, _final_fit_columns(inputs.weekly, cv_spec, config), cv_spec, dissemination,
            model_name=config.final_formula(),
        )
        evaluation = evaluate(inputs.weekly, cv_spec, dissemination, workers=_cv_workers(inputs.weekly, resources))
        inputs.timings["evaluation"] = time.perf_counter() - started
        log.info("evaluation in %.1fs", inputs.timings["evaluation"])
        started = time.perf_counter()
        fitted = final.result()
    inputs.timings["final fit (after the evaluation)"] = time.perf_counter() - started
    blend = TrainedBlend.from_training(config, inputs, fitted)
    return TrainingResult(config=config, inputs=inputs, evaluation=evaluation, blend=blend)


def _final_fit_columns(weekly: pd.DataFrame, cv_spec: dict, config: BlendConfig) -> pd.DataFrame:
    """The weekly table's columns the final fit reads: keys, outcome, the
    climatology weeks and every enabled formula's terms (the sample support
    spans them all); the whole table when a formula names anything else."""
    from ._legacy.blend_evaluation_utils import _parse_formula_cols, build_formulas_from_spec

    formulas = build_formulas_from_spec(
        cv_spec, config.evaluation.cutoff_mode, data=weekly, n_bins=config.features.bins.count
    )
    terms = {column for text in formulas.values() for column in _parse_formula_cols(text)}
    if not terms <= set(weekly.columns):
        return weekly
    keys = {"id", "time", "year", "lat", "lon", "outcome"}
    climatology = f"{cfg.CLIMATOLOGY_FEATURE}_"
    slim = weekly[[c for c in weekly.columns if c in keys or c in terms or c.startswith(climatology)]]
    slim.attrs.update(weekly.attrs)
    return slim


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
