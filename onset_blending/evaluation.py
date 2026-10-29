"""Cross-validated evaluation of the blends and their baselines, on arrays.

The weekly features (`features.Weekly`) and the config in; every model's
out-of-fold probabilities (`core.predictions.Predictions`) and their scores
out, in an `Evaluation`. In order, which is also the row order of the
summary tables:

1. each enabled formula, cross-validated by each CV method (blend.logit:
   global, local, clusters), and the operational fit of the final formula;
2. the random forest blend, when configured (blend.random_forest);
3. external CV predictions;
4. the climatology baselines: unc_clim_raw (the skill baseline), clim_raw
   and its year windows;
5. each model's raw and Platt-calibrated probabilities (core.calibration);
6. the multi-model ensemble (blend.mme).

Models train on the dissemination units and predict every unit; scores
are over the dissemination units (core.metrics). The numerics are those of
the legacy 1_blend_evaluation.py, operation for operation; the artifacts
are NetCDF (predictions, coefficients) and CSV (scores, weights).

Changed from the legacy script: the MME aligns any number of sources on
their samples (id and issue date), where the script's merge failed for two
or more; `evaluation.mme.optimize: always` optimises whatever the holdout
years.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from . import config as cfg
from ._legacy.spatial_id_utils import (
    ensure_spatial_id_col,
    normalize_id_series,
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from .blend import logit, mme, random_forest
from .config import BlendConfig
from .core import calibration, design, metrics
from .core.features import DailyInputs, Weekly
from .core.predictions import Predictions

__all__ = ["Evaluation", "RunSettings", "dissemination_table", "evaluate", "final_rows", "fit_final", "prepare_inputs"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSettings:
    """The years and tags of an evaluation."""

    cutoff_mode: str
    training_years: tuple[int, ...]
    true_holdout_years: tuple[int, ...]
    holdout_years: tuple[int, ...]  # true holdouts, then CV holdouts: the CV folds, in order
    cutoff_tag: str
    year_tag: str

    @classmethod
    def from_config(cls, config: BlendConfig) -> RunSettings:
        years = config.evaluation.years
        return cls(
            cutoff_mode=config.evaluation.cutoff_mode,
            training_years=tuple(years.training),
            true_holdout_years=tuple(years.true_holdout),
            holdout_years=tuple([*years.true_holdout, *years.cv_holdout]),
            cutoff_tag=cfg.cutoff_tag(config),
            year_tag=cfg.year_tag(config),
        )

    @property
    def output_tag(self) -> str:
        """The suffix on every evaluation artifact, e.g. "_fixed_cutoff_2023_2025"."""
        return f"{self.cutoff_tag}{self.year_tag}"

    @property
    def fit_years(self) -> list[int]:
        """The final fit's years: the training years that are not true holdouts."""
        held_out = set(self.true_holdout_years)
        return [year for year in self.training_years if year not in held_out]


def dissemination_table(config: BlendConfig, weekly: Weekly) -> pd.DataFrame:
    """The units the blend trains on and is scored on: the configured table
    (read as strings), or every unit of the weekly features."""
    if config.domain.dissemination is None:
        return pd.DataFrame({"adm3_name": sorted({str(unit) for unit in weekly.units})})
    return pd.read_csv(config.path(config.domain.dissemination.file), dtype=str)


def _id_spec(config: BlendConfig) -> dict:
    """The grid-id convention, as the id utilities read it."""
    ids = config.domain.grid_ids
    if ids.decimal_digits is None:
        return {}
    return {"geometry": {"grid_id_decimal_digits": ids.decimal_digits, "grid_id_format": ids.format}}


def prepare_inputs(weekly: Weekly | pd.DataFrame, config: BlendConfig,
                   dissemination: pd.DataFrame | None = None) -> tuple[Weekly, np.ndarray]:
    """The weekly features with normalised ids (a legacy table is taken as
    a Weekly) and the dissemination unit ids under their convention.

    `dissemination` is the table as read from its CSV (`dtype=str`; default
    `dissemination_table`); its id column is the config's
    `domain.dissemination.id_col`, else `id`, `adm3_name` or lat/lon."""
    spec = _id_spec(config)
    if isinstance(weekly, Weekly):
        # Normalised once per unit: normalisation maps each id on its own.
        ids = normalize_id_series(pd.Series(weekly.units, dtype=object), context="blend input IDs")
        weekly = weekly.with_units(ids.to_numpy(dtype=object))
    else:
        weekly = Weekly.from_frame(ensure_spatial_id_col(weekly, spec=spec, context="blend input IDs"))
    if dissemination is None:
        dissemination = dissemination_table(config, weekly)
    convention = resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=pd.Series(weekly.units, dtype=object),
        context="blend input/dissemination ID handoff",
    )
    id_col = config.domain.dissemination.id_col if config.domain.dissemination is not None else None
    if id_col:
        if id_col not in dissemination.columns:
            raise ValueError(f"cell.id_col '{id_col}' not found. Found: {dissemination.columns.tolist()}")
        dissemination = dissemination.rename(columns={id_col: "id"})
    cells = ensure_spatial_id_col(dissemination, spec=spec, convention=convention, context="blend dissemination IDs")
    validate_id_coordinate_consistency(cells, context="blend dissemination IDs")
    weeks = _present_weeks(weekly.columns, cfg.CLIMATOLOGY_FEATURE)
    n_bins = config.features.bins.count
    if weeks != list(range(1, n_bins + 1)):
        raise ValueError(
            f"The weekly features have climatology weeks {weeks} "
            f"({cfg.CLIMATOLOGY_FEATURE}_week<n>); the config bins {n_bins}."
        )
    return weekly, np.asarray(pd.unique(cells["id"]), dtype=object)


def log_sample_support(support: dict) -> None:
    """Log the rows the formulas' common sample support kept, as the scripts printed it."""
    log.info(
        "Formula sample support (%s): %s -> %s rows (%s excluded)",
        support["policy"], support["rows_before"], support["rows_after"], support["rows_excluded"],
    )
    if support["nonfinite_by_column"]:
        log.info("Non-finite formula predictors: %s", support["nonfinite_by_column"])


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    """Every model's CV predictions and scores, and the fitted weights.

    `save` writes them under the legacy script's artifact names, as NetCDF
    (predictions, coefficients), CSV (scores, weights) and JSON."""

    output_tag: str
    sample_support: dict  # formula_sample_support{tag}.json
    formulas: dict[str, str]  # name -> the resolved formula that was cross-validated
    cell_metrics: pd.DataFrame | None = None  # summary_models{tag}
    pooled: pd.DataFrame | None = None  # summary_models_pooled{tag}
    yearly: pd.DataFrame | None = None  # yearly_metrics_global{tag}
    cv_preds: dict[tuple[str, str], Predictions] = field(default_factory=dict)  # (model, method)
    coefficients: dict[tuple[str, str], dict[int, logit.FittedModel]] = field(default_factory=dict)  # by holdout year
    platt_weights: dict[str, pd.DataFrame] = field(default_factory=dict)  # file stem -> weights
    mme_weights: dict[str, pd.DataFrame] = field(default_factory=dict)  # file stem -> weights
    final_model: logit.FittedModel | None = None  # evaluate(final=True); saved in the trained blend
    final_forest: random_forest.FittedForest | None = None  # likewise

    def save(self, directory: str | os.PathLike, *, pickles: bool = False) -> list[Path]:
        """Write the artifacts into `directory` (pickles: also .pkl twins of
        the score tables); returns the paths written."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tag = self.output_tag
        written = []

        def wrote(path):
            written.append(path)
            log.info("Wrote %s", path)

        path = directory / f"formula_sample_support{tag}.json"
        path.write_text(json.dumps(self.sample_support, indent=2))
        wrote(path)
        for (model, method), preds in self.cv_preds.items():
            path = directory / f"cv_preds_{model}_{method}{tag}.nc"
            preds.to_netcdf(path)
            wrote(path)
            folds = self.coefficients.get((model, method))
            if folds:
                path = directory / f"coefs_{model}_{method}{tag}.nc"
                logit.fold_coefficients(folds).to_netcdf(path)
                wrote(path)
        for stem, weights in {**self.platt_weights, **self.mme_weights}.items():
            path = directory / f"{stem}.csv"
            weights.to_csv(path, index=False)
            wrote(path)
        tables = [(f"summary_models{tag}", self.cell_metrics, True), (f"summary_models_pooled{tag}", self.pooled, False),
                  (f"yearly_metrics_global{tag}", self.yearly, True)]
        for stem, table, twin in tables:
            if table is None:
                continue
            path = directory / f"{stem}.csv"
            table.to_csv(path, index=False)
            wrote(path)
            if pickles and twin:
                path = directory / f"{stem}.pkl"
                table.to_pickle(path)
                wrote(path)
        return written


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


def evaluate(
    weekly: Weekly | pd.DataFrame,
    config: BlendConfig,
    dissemination: pd.DataFrame | None = None,
    *,
    workers: int = 1,
    results_dir: str | os.PathLike | None = None,
    budget_bytes: int | None = None,
    final: bool = False,
    forest_inputs: DailyInputs | None = None,
) -> Evaluation:
    """Cross-validate the config's blends and score its baselines on `weekly`
    (a `features.Weekly`, or the legacy table).

    `dissemination` is the dissemination table as read from its CSV (see
    `prepare_inputs`). Up to `workers` processes fit the global CV folds,
    as many as `budget_bytes` holds, and the Platt calibration; results do
    not depend on either. `results_dir` is where relative
    `external_predictions` files and saved MME weights are read from.
    `final` also fits the final formula (and the forest) on every training
    year, into `final_model` (`final_forest`): the formula alongside the
    global CV folds, on their design matrix. The random forest needs
    `forest_inputs` (features.daily_inputs, for these weekly samples).

    Formula and CV errors raise; climatology, forecast and MME source
    problems warn and skip that model, as the script did.
    """
    if workers < 1:
        raise ValueError("workers must be a positive integer.")
    run = RunSettings.from_config(config)
    weekly, cells = prepare_inputs(weekly, config, dissemination)
    labels = cfg.outcome_labels(config)
    externals = _load_external_predictions(config.evaluation.external_predictions, results_dir, labels)

    # Formulas come before sample support: only predictors that enabled
    # formulas reference may decide the common evaluation rows.
    formulas, resolution = design.formulas(
        config.evaluation, weekly.attrs.get("unavailable_rain_predictors", {}), len(labels) - 1, run.cutoff_mode
    )
    rows, support = design.sample_support(weekly, formulas)
    support["formula_resolution"] = resolution
    log_sample_support(support)
    log.info("Formulas: %s", list(formulas))

    ctx = _Context(config, run, weekly, rows, cells, labels, budget_bytes=budget_bytes)
    result = Evaluation(output_tag=run.output_tag, sample_support=support, formulas=formulas)
    final_name = config.final_formula() if final else None
    train_final = final_rows(ctx) if final else None
    scores = _Scores(ctx)
    _cross_validate(ctx, formulas, workers, result, scores, final_name)
    forest = config.evaluation.random_forest
    if forest is not None:
        if forest_inputs is None:
            raise ValueError("the random forest blend needs its inputs (features.daily_inputs)")
        params = random_forest.Params.from_config(forest, config.features.bins.days)
        _forest(ctx, params, forest_inputs, workers, result, scores, with_final=final)
    if final and result.final_model is None:
        log.info("\nFitting %s on all %d training years...", final_name, len(run.fit_years))
        result.final_model = logit.fit_samples(formulas[final_name], weekly, train_final)
    # After the CV, the Platt calibration fans out over worker processes
    # (with one BLAS thread each).
    with _scoring_pool(workers) as pool:
        ctx.pool = pool
        for external, preds in externals:
            log.info("\nExternal CV predictions: %s (%s)", external.name, external.file)
            scores.add(preds, external.name, external.method, yearly=True)
        clim_preds = _climatology_baselines(ctx, scores)
        calibrated_preds = _forecast_baselines(ctx, result, scores)
        _mme(ctx, clim_preds, calibrated_preds, results_dir, result, scores)

    if scores.cells:
        result.cell_metrics = pd.concat(scores.cells, ignore_index=True)
        try:
            result.pooled = metrics.pooled_skill(result.cell_metrics)
        except Exception as e:
            warnings.warn(f"Could not compute pooled summary: {e}")
    if scores.yearly:
        result.yearly = pd.DataFrame(scores.yearly)
    return result


def fit_final(weekly: Weekly | pd.DataFrame, config: BlendConfig,
              dissemination: pd.DataFrame | None = None) -> logit.FittedModel:
    """The final formula fitted on every training year that is not a true
    holdout, on the dissemination units, after the evaluation's sample
    support; without the CV (`evaluate(..., final=True)` fits it alongside)."""
    run = RunSettings.from_config(config)
    weekly, cells = prepare_inputs(weekly, config, dissemination)
    labels = cfg.outcome_labels(config)
    formulas, _ = design.formulas(
        config.evaluation, weekly.attrs.get("unavailable_rain_predictors", {}), len(labels) - 1, run.cutoff_mode
    )
    rows, support = design.sample_support(weekly, formulas)
    log_sample_support(support)
    name = config.final_formula()
    train = final_rows(_Context(config, run, weekly, rows, cells, labels))
    log.info("\nFormula : %s", formulas[name])
    log.info("\nFitting %s on all %d training years...", name, len(run.fit_years))
    return logit.fit_samples(formulas[name], weekly, train)


@dataclass
class _Context:
    config: BlendConfig
    run: RunSettings
    weekly: Weekly  # every sample
    rows: np.ndarray  # the formula-supported samples, of every unit
    cells: np.ndarray  # the dissemination unit ids
    labels: list[str]  # the outcome bins: week1..weekN, later
    budget_bytes: int | None = None
    pool: object = None  # the Platt pool, once the CV is done

    @functools.cached_property
    def year(self) -> np.ndarray:
        return self.weekly.take("year", self.rows)

    @functools.cached_property
    def unit_allowed(self) -> np.ndarray:
        """Per unit of the weekly features: a dissemination unit."""
        cells = set(self.cells.tolist())
        return np.fromiter((unit in cells for unit in self.weekly.units), dtype=bool, count=len(self.weekly.units))

    @functools.cached_property
    def allowed(self) -> np.ndarray:
        """Per supported sample: in a dissemination unit (models train on those)."""
        return self.unit_allowed[self.weekly.sample_unit[self.rows]]

    def in_years(self, years) -> np.ndarray:
        """The supported samples in these years (a mask over rows)."""
        return np.isin(self.year, list(years))

    def allowed_ids(self, ids: np.ndarray) -> np.ndarray:
        """Per id: a dissemination unit."""
        cells = set(self.cells.tolist())
        return np.fromiter((i in cells for i in ids.tolist()), dtype=bool, count=len(ids))


def final_rows(ctx: _Context) -> np.ndarray:
    """The final fit's samples (positions): the supported samples in the
    dissemination units and the fit years, with every outcome class."""
    fit_years = ctx.run.fit_years
    candidates = ctx.rows[ctx.allowed]
    train = candidates[np.isin(ctx.weekly.take("year", candidates), fit_years)]
    if not len(train):
        raise ValueError("Final fit has no rows in the configured training years and dissemination cells.")
    present = {value for value in pd.unique(ctx.weekly.take("outcome", train)) if isinstance(value, str)}
    missing = [value for value in ctx.labels if value not in present]
    if missing:
        raise ValueError("Final-fit training rows are missing outcome classes: " + ", ".join(missing))
    log.info("  Training years        : %s", sorted(fit_years))
    log.info("  Training rows (allowed cells) : %d", len(train))
    return train


@contextlib.contextmanager
def _scoring_pool(workers: int):
    """Spawned processes with one BLAS thread each, or None for one worker."""
    if workers <= 1:
        yield None
        return
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from .core.parallel import limit_blas_threads

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=limit_blas_threads, initargs=(1,)
    ) as pool:
        yield pool


@dataclass
class _Scores:
    """Per-unit metric frames and yearly metric rows, in the order models are scored."""

    ctx: _Context
    cells: list = field(default_factory=list)
    yearly: list = field(default_factory=list)

    def add(self, preds: Predictions, model: str, method: str, *, yearly: bool) -> None:
        """Per-unit metrics over the dissemination units; with yearly (CV
        predictions), per-year metrics over every unit."""
        P, outcome = preds.probabilities, preds.outcome
        cells = metrics.cell_metrics(
            P, outcome, preds.ids, preds.labels, allowed=self.ctx.allowed_ids(preds.ids), earlier=preds.earlier
        )
        cells["model"] = model
        cells["cv_method"] = method
        self.cells.append(cells)
        if yearly:
            self.yearly.extend(metrics.yearly_metrics(P, outcome, preds.year, model, method))


# ---------------------------------------------------------------------------
# 1-3: the blends and external predictions
# ---------------------------------------------------------------------------


def _cv_methods(config: BlendConfig) -> list[str]:
    """The CV methods, under the names the artifacts carry."""
    return ["cluster" if m == "clusters" else m for m in config.evaluation.cv_methods]


def _cross_validate(ctx: _Context, formulas: dict[str, str], workers: int, result: Evaluation,
                    scores: _Scores, final_model: str | None = None) -> None:
    """Each formula under each CV method; with the global method, also the
    final fit of `final_model`."""
    run = ctx.run
    dtypes = ctx.weekly.column_dtypes()
    for model, formula in formulas.items():
        log.info("\nCV: %s", model)
        for method in _cv_methods(ctx.config):
            log.info("  Method: %s", method)
            try:
                folds = {}
                if method == "global":
                    plan = design.plan(formula, dtypes)
                    if plan is not None:
                        parts, fitted = _global_cv(ctx, plan, workers, model == final_model)
                        if fitted is not None:
                            result.final_model = fitted
                    else:
                        parts = logit.global_cv_frames(
                            formula, ctx.weekly, ctx.rows, ctx.allowed,
                            holdout_years=run.holdout_years, training_years=run.training_years,
                            true_holdout_years=run.true_holdout_years, required_classes=ctx.labels,
                        )
                        if not parts:
                            raise ValueError("Global CV produced no fold predictions.")
                    folds = {year: fold_model for year, _, _, fold_model in parts}
                    preds = _fold_predictions(ctx, [(test, P, m.classes) for _, test, P, m in parts])
                elif method == "local":
                    preds = _fold_predictions(ctx, logit.local_cv(formula, ctx.weekly, ctx.rows, run.holdout_years))
                elif method == "cluster":
                    cluster_var = ctx.config.evaluation.cluster_var
                    parts = logit.cluster_cv(formula, ctx.weekly, ctx.rows, run.holdout_years, cluster_var)
                    preds = _fold_predictions(ctx, parts)
                else:
                    warnings.warn(f"Unknown CV method: {method}. Skipping.")
                    continue
                if preds is None or not len(preds):
                    raise ValueError(f"No predictions for {model}/{method}.")
                scores.add(preds, model, method, yearly=True)
                result.cv_preds[(model, method)] = preds
                if folds:
                    result.coefficients[(model, method)] = folds
            except Exception as e:
                raise RuntimeError(f"Error in {model}/{method}: {e}") from e


def _fold_predictions(ctx: _Context, parts) -> Predictions | None:
    """CV folds' predictions, [(positions into rows, probabilities, classes)],
    as Predictions over the outcome bins (NaN for a class a fold lacked)."""
    if not parts:
        return None
    index = {label: j for j, label in enumerate(ctx.labels)}
    positions = np.concatenate([test for test, _, _ in parts])
    P = np.full((len(positions), len(ctx.labels)), np.nan)
    start = 0
    for test, probabilities, classes in parts:
        for k, label in enumerate(classes):
            if label in index:
                P[start:start + len(test), index[label]] = probabilities[:, k]
        start += len(test)
    return Predictions.of_samples(ctx.weekly, ctx.rows[positions], P, ctx.labels)


def _global_cv(ctx: _Context, plan: design.Plan, workers: int, with_final: bool):
    """Global CV on the shared design: (folds as global_cv returns them, the
    final model or None)."""
    run = ctx.run
    folds, final = logit.global_cv(
        plan,
        ctx.weekly.take,
        ctx.rows,
        ctx.allowed,
        ctx.year,
        ctx.weekly.take("outcome", ctx.rows),
        holdout_years=run.holdout_years,
        training_years=run.training_years,
        true_holdout_years=run.true_holdout_years,
        required_classes=ctx.labels,
        final_years=run.fit_years if with_final else None,
        workers=workers,
        budget_bytes=ctx.budget_bytes,
    )
    if not folds:
        raise ValueError("Global CV produced no fold predictions.")
    fitted = None
    if final is not None:
        clf, scaler, used, total = final
        logit.log_fit(used, total, clf.classes_, plan.columns)
        fitted = logit.FittedModel.from_estimators(plan.formula, plan.columns, clf, scaler)
    return folds, fitted


def _forest(ctx: _Context, params: random_forest.Params, inputs: DailyInputs, workers: int,
            result: Evaluation, scores: _Scores, *, with_final: bool) -> None:
    """The random forest blend: out-of-fold weekly probabilities for the
    samples the formulas score, from forests fitted on every sample of the
    dissemination units in each fold's training years; then the final fit."""
    run = ctx.run
    weekly = ctx.weekly
    models = inputs.model_names
    X = random_forest.feature_matrix(weekly, inputs, np.arange(len(weekly)), models)
    params = replace(params, n_jobs=random_forest.workers(len(X), workers, ctx.budget_bytes))
    names = random_forest.feature_names(inputs, models)
    year = weekly["year"]
    trainable = ctx.unit_allowed[weekly.sample_unit]
    scoreable = np.zeros(len(weekly), dtype=bool)
    scoreable[ctx.rows] = True
    true_holdouts = set(run.true_holdout_years)
    fold_years = random_forest.folds(
        run.training_years,
        [y for y in run.holdout_years if y not in true_holdouts],
        run.true_holdout_years,
        params.years_per_fold,
    )
    log.info("\nCV: %s (%d folds)", params.name, len(fold_years))
    folds = random_forest.cross_validate(X, inputs.onset_lead, names, year, trainable, scoreable, fold_years, params)
    test = np.concatenate([rows for rows, _ in folds])
    daily = np.concatenate([probabilities for _, probabilities in folds])
    weekly_probs = random_forest.weekly_from_daily(daily, params.days_per_week)
    preds = Predictions.of_samples(weekly, test, weekly_probs, ctx.labels, daily=daily)
    scores.add(preds, params.name, "global", yearly=True)
    result.cv_preds[(params.name, "global")] = preds
    if with_final:
        train = np.flatnonzero(trainable & np.isin(year, run.fit_years))
        log.info("\nFitting %s on all %d training years...", params.name, len(run.fit_years))
        result.final_forest = random_forest.fit(X[train], inputs.onset_lead[train], names, params)


def _load_external_predictions(externals, results_dir, labels) -> list[tuple[cfg.ExternalPrediction, Predictions]]:
    """evaluation.external_predictions, loaded: NetCDF predictions, or a
    pickled legacy table; relative files resolve against results_dir."""
    if results_dir is None:
        relative = [e.file for e in externals if not os.path.isabs(e.file)]
        if relative:
            raise ValueError(
                f"extras.external_predictions files {relative} are relative, "
                "but no results_dir was given to resolve them against."
            )
        results_dir = ""
    return [(external, _external(external, os.path.join(results_dir, external.file), labels)) for external in externals]


def _external(external: cfg.ExternalPrediction, path: str, labels) -> Predictions:
    where = f"External predictions '{external.name}' ({path})"
    if path.endswith(".nc"):
        preds = Predictions.open(path)
        if preds.labels != list(labels):
            raise ValueError(f"{where}: bins {preds.labels}, expected {list(labels)}.")
        if not len(preds):
            raise ValueError(f"{where} has no rows.")
        return preds
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{where} must be a pandas DataFrame, got {type(frame).__name__}.")
    required = ["id", "year", "outcome"] + [f"cv_{b}" for b in labels]
    missing = [c for c in required if c not in frame.columns]
    n_bins = len(labels) - 1
    unexpected = [f"cv_week{w}" for w in _present_weeks(frame.columns, "cv") if w > n_bins]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing columns {missing}")
        if unexpected:
            problems.append(f"unexpected columns {unexpected} beyond {n_bins} week bins")
        raise ValueError(f"{where}: {'; '.join(problems)}. Expected {required}.")
    if frame.empty:
        raise ValueError(f"{where} has no rows.")
    return Predictions.from_frame(frame.reset_index(drop=True), labels)


# ---------------------------------------------------------------------------
# 4-5: climatology and forecast baselines
# ---------------------------------------------------------------------------


def _present_weeks(columns, base: str) -> list[int]:
    """Sorted week numbers present as `<base>_week<n>` columns."""
    pattern = re.compile(rf"^{re.escape(base)}_week(\d+)$")
    return sorted(int(m.group(1)) for c in columns for m in [pattern.match(c)] if m)


def _row_sum(columns: list[np.ndarray]) -> np.ndarray:
    """Per row, the sum of the columns, a missing value counting as 0,
    added left to right (as the legacy table's row sums are)."""
    total = np.where(np.isnan(columns[0]), 0.0, columns[0])
    for values in columns[1:]:
        total = total + np.where(np.isnan(values), 0.0, values)
    return total


def _week_columns(ctx: _Context, rows: np.ndarray, base: str, *, logits: bool):
    """(the week labels present as `<base>_week<n>`, their probabilities for rows)."""
    weeks = _present_weeks(ctx.weekly.columns, base)
    values = [np.asarray(ctx.weekly.take(f"{base}_week{w}", rows), dtype=float) for w in weeks]
    if logits:
        values = [expit(v) for v in values]
    return [f"week{w}" for w in weeks], values


def _holdout_rows(ctx: _Context) -> np.ndarray:
    return ctx.rows[ctx.in_years(ctx.run.holdout_years)]


def _climatology_models(config: BlendConfig) -> list[tuple[str, str]]:
    """(model, column prefix) of each climatology baseline, in scoring order."""
    feature = cfg.CLIMATOLOGY_FEATURE
    baselines = config.evaluation.climatology_baselines
    out = [("unc_clim_raw", f"{feature}_unc")] if baselines.unconditional else []
    if baselines.conditional:
        out.append(("clim_raw", feature))
    out += [(f"clim_raw_{cfg.window_tag(w)}", f"{feature}_{cfg.window_tag(w)}") for w in baselines.windows]
    return out


def _climatology_baselines(ctx: _Context, scores: _Scores) -> dict[str, Predictions]:
    """The climatology logits as probabilities for the holdout years. Returns
    the predictions by model name, for the MME."""
    rows = _holdout_rows(ctx)
    out = {}
    for model, prefix in _climatology_models(ctx.config):
        try:
            labels, weeks = _week_columns(ctx, rows, prefix, logits=True)
            if not labels:
                warnings.warn(f"Skipping clim logit raw for prefix={prefix}; no _week* columns found")
                continue
            if model == "unc_clim_raw":
                earlier_col = f"{prefix}_earlier"
                if earlier_col not in ctx.weekly.columns:
                    raise ValueError("unc_clim_raw requires earlier_col and it must exist in wide_df.")
                earlier = np.clip(expit(np.asarray(ctx.weekly.take(earlier_col, rows), dtype=float)), 0.0, 1.0)
                later = np.clip(1.0 - (earlier + _row_sum(weeks)), 0.0, 1.0)
                # The earlier bin is renormalised with the rest.
                every = [earlier, *weeks, later]
                total = _row_sum(every)
                good = total > 0
                earlier, *weeks, later = [np.where(good, v / np.where(good, total, 1.0), v) for v in every]
            else:
                earlier = None
                later = np.clip(1.0 - _row_sum(weeks), 0.0, 1.0)
            if not len(rows):
                continue
            preds = Predictions.of_samples(ctx.weekly, rows, np.column_stack([*weeks, later]), labels + ["later"],
                                           earlier=earlier)
            out[model] = preds
            scores.add(preds, model, "raw", yearly=False)
        except Exception as e:
            warnings.warn(f"Error in clim logit '{model}': {e}")
    return out


def _cutoff_variant(config: BlendConfig) -> str:
    return {"fixed_cutoff": "fixed_cutoff", "ref_onset": "ref_onset"}.get(config.evaluation.cutoff_mode, "base")


def _forecast_baselines(ctx: _Context, result: Evaluation, scores: _Scores) -> dict[str, Predictions]:
    """Each model's probabilities, raw and Platt-calibrated. Returns the
    calibrated predictions by MME key."""
    run = ctx.run
    calibrated_by_key = {}
    for scoring in ctx.config.forecast_evaluations():
        name = scoring.model
        variant = scoring.variant or _cutoff_variant(ctx.config)
        label = name if variant == "base" else f"{name}_{variant}"
        base = cfg.probability_feature_base(name, variant)
        if scoring.raw:
            try:
                rows = _holdout_rows(ctx)
                labels, weeks = _week_columns(ctx, rows, base, logits=False)
                if not labels:
                    warnings.warn(f"Skipping raw for {name} ({variant}); no {base}_week* columns found")
                elif len(rows):
                    later = np.clip(1.0 - _row_sum(weeks), 0.0, 1.0)
                    preds = Predictions.of_samples(ctx.weekly, rows, np.column_stack([*weeks, later]), labels + ["later"])
                    scores.add(preds, f"{label}_raw", "raw", yearly=False)
            except Exception as e:
                warnings.warn(f"Error in raw forecast '{name}': {e}")

        if scoring.calibrated:
            try:
                preds, weights = _calibrated(ctx, base, name, variant, export=scoring.platt_weights)
                if preds is not None and len(preds):
                    key = name if variant == "base" else f"{name}_calibrated_{variant}"
                    calibrated_by_key[key] = preds
                    scores.add(preds, f"{name}_calibrated" if variant == "base" else key, "calibrated", yearly=False)
                    if weights is not None:
                        result.platt_weights[f"platt_weights_{label}_calibrated_df{run.output_tag}"] = weights
            except Exception as e:
                warnings.warn(f"Error in calibrated forecast '{name}': {e}")
    return calibrated_by_key


def _calibrated(ctx: _Context, base: str, name: str, variant: str, *, export: bool):
    """Platt-calibrated CV predictions for the holdout years, from fits on the
    training and holdout years' samples (outside the true holdouts, in the
    dissemination units); with export, the weights fitted on the training
    years' samples. (None, None) without the model's columns."""
    run = ctx.run
    rows = ctx.rows[ctx.in_years([*run.training_years, *run.holdout_years])]
    weeks, values = _week_columns(ctx, rows, base, logits=False)
    if not weeks:
        warnings.warn(f"Skipping calibrated for {name} ({variant}); no {base}_week* columns found")
        return None, None
    labels = weeks + ["later"]
    P = np.column_stack([*values, np.maximum(0.0, 1.0 - _row_sum(values))])
    outcome = ctx.weekly.outcome_codes(rows, labels)
    year = ctx.weekly.take("year", rows)
    allowed = ctx.allowed[ctx.in_years([*run.training_years, *run.holdout_years])]
    trainable = ~np.isin(year, run.true_holdout_years) & allowed
    folds = calibration.cross_validated(P, outcome, year, trainable, run.holdout_years, executor=ctx.pool)
    preds = None
    if folds:
        positions = np.concatenate([test for _, test, _ in folds])
        preds = Predictions.of_samples(ctx.weekly, rows[positions], np.concatenate([c for _, _, c in folds]), labels)
    weights = None
    if export and preds is not None:
        train = np.isin(year, run.training_years)
        weights = calibration.export_weights(P[train], outcome[train], labels)
    return preds, weights


# ---------------------------------------------------------------------------
# 6: the multi-model ensemble
# ---------------------------------------------------------------------------


def _mme(ctx: _Context, clim_preds, calibrated_preds, results_dir, result: Evaluation, scores: _Scores) -> None:
    """The MME of climatology and calibrated forecasts, with RPS-optimised
    weights, or saved ones (see config.Mme.optimize)."""
    settings = ctx.config.evaluation.mme
    if not (settings.enabled and clim_preds):
        return
    run = ctx.run
    names = [source.name for source in settings.sources]
    weight_cols = [f"w_{name}" for name in names]
    optimise = settings.optimize == "always" or (
        settings.optimize == "auto" and sorted(run.holdout_years) == list(range(2000, 2025))
    )
    clim_model = next((s.name for s in settings.sources if s.source == "clim"), names[0])
    for variant in settings.variants:
        log.info("\n=== MME optimization (RPS-only) variant: %s ===", variant)
        sources = _mme_sources(settings.sources, clim_preds, calibrated_preds)
        if sources is None:
            continue
        P_full, common = mme.align(sources)
        if any(source.labels != ctx.labels for source in sources):
            warnings.warn("MME sources must all have the outcome bins as their labels. Skipping.")
            continue
        base = sources[0]
        ids = base.ids[common]
        allowed = ctx.allowed_ids(ids)
        P_allowed = [metrics.clean(P[allowed]) for P in P_full]
        Y_allowed = metrics.one_hot(base.outcome[common][allowed], len(ctx.labels))

        stem = f"mme_optimized_weights{run.cutoff_tag}_{clim_model}_{variant}"
        if optimise:
            log.info("  Running MME optimization...")
            weights, rps = mme.optimise_weights(P_allowed, Y_allowed)
            result.mme_weights[f"{stem}{run.year_tag}"] = pd.DataFrame(
                [{"objective": "rps", **dict(zip(weight_cols, weights)), "rps": rps}]
            )
        else:
            log.info("  Reading saved MME weights...")
            saved = _read_mme_weights(results_dir, f"{stem}{settings.weights_year_tag or run.year_tag}.csv")
            if saved is None:
                continue
            row = saved[saved["objective"] == "rps"].iloc[0]
            weights = np.array([row[w] for w in weight_cols])
            rps = mme.mixed_rps(weights, P_allowed, Y_allowed)

        log.info("  MME weights: %s", dict(zip(names, np.round(weights, 4))))
        log.info("  RPS (pooled, allowed): %.6f", rps)
        preds = base.take(common).with_probabilities(mme.mix(weights, P_full))
        model = f"mme_{variant}_{clim_model}_opt_rps"
        scores.add(preds, model, "mme", yearly=False)
        log.info("  Added MME model: %s", model)


def _mme_sources(sources, clim_preds, calibrated_preds) -> list[Predictions] | None:
    """Each source's CV predictions, in order; None, with a warning, if any is missing."""
    out = []
    for source in sources:
        if source.source == "clim":
            preds = clim_preds.get(source.name)
            if preds is None:
                warnings.warn(f"MME blend_model '{source.name}' (source: clim) not found.")
                return None
        else:
            key = source.name if source.cal_variant == "base" else f"{source.name}_calibrated_{source.cal_variant}"
            preds = calibrated_preds.get(key)
            if preds is None:
                warnings.warn(f"MME blend_model '{source.name}' (key: {key}) not found.")
                return None
        out.append(preds)
    return out


def _read_mme_weights(results_dir, name) -> pd.DataFrame | None:
    """Saved MME weights (CSV) from results_dir; None, with a warning, if absent."""
    if results_dir is None:
        warnings.warn(f"MME weights file not found: {name} (no results_dir given). Skipping.")
        return None
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        warnings.warn(f"MME weights file not found: {path}. Skipping.")
        return None
    return pd.read_csv(path)
