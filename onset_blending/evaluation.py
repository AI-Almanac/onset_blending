"""Cross-validated evaluation of the blend formulas and their baselines.

A port of the body of 1_blend_evaluation.py (phase 2 of
docs/pipeline-redesign.md): the same steps and numerics, with the weekly
features, the cv_models spec and the dissemination table passed in, and
every artifact the script writes returned in an `Evaluation`. Results are
bit-identical to the script's; CV predictions keep only their keys
(id, time, year, lat, lon, outcome) beside the probabilities.

The weekly features stay arrays (`features.Weekly`): the formulas' sample
support and the dissemination cells are positions into them, global CV
fits on one shared design matrix (`cv`), and the baselines read slim
tables of just the rows and columns they use.

The steps, in the script's order (which is also the row order of the summary
tables): cross-validate each enabled formula with each CV method; score
external CV predictions; climatology-logit baselines; raw and Platt-calibrated
forecasts; the MME blend.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .blend import logit
from .core import design
from ._legacy.blend_evaluation_utils import (
    _present_weeks,
    build_formulas_from_spec,
    clean_probs5,
    compute_cell_metrics_fast,
    compute_cv_clusters,
    compute_cv_global,
    compute_cv_local,
    compute_cv_neighbors,
    configured_holdout_years,
    fit_platt_weights_export,
    forecast_label,
    formula_sample_support,
    get_forecast_variant_suffix,
    load_external_predictions,
    make_calibrated_preds_from_wide,
    make_cutoff_tag,
    make_raw_preds_from_wide,
    make_raw_preds_from_wide_logit,
    make_raw_preds_from_wide_logit_window,
    make_year_tag,
    optimize_mme_weights,
    parse_external_predictions,
    pooled_rps5,
    resolve_climatology_weeks,
    restrict_to_allowed,
    score_model_predictions,
    summarize_models_pooled,
)
from ._legacy.misc import coalesce, interval_bins
from ._legacy.spatial_id_utils import (
    ensure_spatial_id_col,
    normalize_id_series,
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from .core.features import Weekly

__all__ = ["Evaluation", "RunSettings", "evaluate", "prepare_inputs", "sample_support"]

# The columns CV predictions keep beside their probabilities.
PREDICTION_KEYS = ("id", "time", "year", "lat", "lon", "outcome")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSettings:
    """The cv spec's `run` section, with the scripts' defaults applied."""

    cutoff_mode: str
    training_years: tuple[int, ...]
    true_holdout_years: tuple[int, ...]
    holdout_years: tuple[int, ...]  # true holdouts, then CV holdouts: the CV folds, in order

    @classmethod
    def from_spec(cls, spec: dict) -> RunSettings:
        run = spec.get("run", {})
        return cls(
            cutoff_mode=coalesce(run.get("cutoff_mode"), "ref_onset"),
            training_years=tuple(map(int, coalesce(run.get("training_years"), range(2000, 2025)))),
            true_holdout_years=tuple(map(int, coalesce(run.get("true_holdout_years"), []))),
            holdout_years=tuple(configured_holdout_years(spec)),
        )

    @property
    def cutoff_tag(self) -> str:
        return make_cutoff_tag(self.cutoff_mode)

    @property
    def year_tag(self) -> str:
        return make_year_tag(self.holdout_years)

    @property
    def output_tag(self) -> str:
        """The suffix on every evaluation artifact, e.g. "_fixed_cutoff_2023_2025"."""
        return f"{self.cutoff_tag}{self.year_tag}"

    @property
    def fit_years(self) -> list[int]:
        """The final fit's years: the training years that are not true holdouts."""
        held_out = set(self.true_holdout_years)
        return [year for year in self.training_years if year not in held_out]


def prepare_inputs(
    weekly: Weekly | pd.DataFrame, cv_spec: dict, dissemination: pd.DataFrame
) -> tuple[Weekly, pd.DataFrame, int]:
    """The weekly features with normalised ids (a legacy table is taken as
    a Weekly), the dissemination cells under their id convention, and the
    number of week bins.

    `dissemination` is the table as read from its CSV (`dtype=str`); its id
    column is `cell.id_col` (or `cell.dissemination_id_col`) when the spec
    names one, else `id`, `adm3_name` or lat/lon.
    """
    if isinstance(weekly, Weekly):
        # Normalised once per unit: normalisation maps each id on its own.
        ids = normalize_id_series(pd.Series(weekly.units, dtype=object), context="blend input IDs")
        weekly = weekly.with_units(ids.to_numpy(dtype=object))
    else:
        weekly = Weekly.from_frame(ensure_spatial_id_col(weekly, spec=cv_spec, context="blend input IDs"))
    convention = resolve_grid_id_convention(
        spec=cv_spec,
        authoritative_ids=pd.Series(weekly.units, dtype=object),
        context="blend input/dissemination ID handoff",
    )
    cell_cfg = cv_spec.get("cell") or {}
    id_col = cell_cfg.get("id_col") or cell_cfg.get("dissemination_id_col")
    if id_col:
        if id_col not in dissemination.columns:
            raise ValueError(
                f"cell.id_col '{id_col}' not found. Found: {dissemination.columns.tolist()}"
            )
        dissemination = dissemination.rename(columns={id_col: "id"})
    cells = ensure_spatial_id_col(
        dissemination, spec=cv_spec, convention=convention, context="blend dissemination IDs"
    )
    validate_id_coordinate_consistency(cells, context="blend dissemination IDs")
    # The week bins are the configured climatology columns the connect stage wrote.
    _, clim_weeks = resolve_climatology_weeks(cv_spec, weekly)
    return weekly, cells.drop_duplicates("id"), max(clim_weeks)


def sample_support(weekly: Weekly, formulas: dict[str, str]) -> tuple[np.ndarray, dict]:
    """The samples every enabled formula's predictors are finite for (as
    positions) and the diagnostics apply_formula_sample_support gives."""
    keep, diagnostics = formula_sample_support(weekly, formulas)
    return np.flatnonzero(keep), diagnostics


def in_cells(weekly: Weekly, cells: pd.DataFrame) -> np.ndarray:
    """Per sample: whether its unit is one of the cells (restrict_to_allowed)."""
    ids = set(cells["id"])
    return np.fromiter((unit in ids for unit in weekly.units), dtype=bool, count=len(weekly.units))[weekly.sample_unit]


def log_sample_support(support: dict) -> None:
    """Log the rows the formulas' common sample support kept, as the scripts printed it."""
    log.info(
        "Formula sample support (%s): %s -> %s rows (%s excluded)",
        support["policy"],
        support["rows_before"],
        support["rows_after"],
        support["rows_excluded"],
    )
    if support["nonfinite_by_column"]:
        log.info("Non-finite formula predictors: %s", support["nonfinite_by_column"])


@dataclass
class Evaluation:
    """Everything 1_blend_evaluation.py writes, in memory.

    Keys and file stems follow the script's artifact names; `save` writes
    them under exactly those names.
    """

    output_tag: str
    sample_support: dict  # formula_sample_support{tag}.json
    formulas: dict[str, str]  # name -> the resolved formula that was cross-validated
    cell_metrics: pd.DataFrame | None = None  # summary_models{tag}
    pooled: pd.DataFrame | None = None  # summary_models_pooled{tag}
    yearly: pd.DataFrame | None = None  # yearly_metrics_global{tag}
    # (model, method) -> the PREDICTION_KEYS and cv_<class> probabilities
    cv_preds: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)
    coefs_by_year: dict[tuple[str, str], dict[int, dict]] = field(default_factory=dict)
    platt_weights: dict[str, pd.DataFrame] = field(default_factory=dict)  # file stem -> weights
    mme_weights: dict[str, pd.DataFrame] = field(default_factory=dict)  # file stem -> weights
    final_model: object = None  # blend.logit.FittedModel, when evaluate() was asked for it (not saved here)

    def save(self, directory: str | os.PathLike) -> list[Path]:
        """Write the script's files into `directory`; returns the paths written."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tag = self.output_tag
        written = []

        def write(name, obj):
            path = directory / name
            if name.endswith(".json"):
                with path.open("w") as handle:
                    json.dump(obj, handle, indent=2)
            elif name.endswith(".csv"):
                obj.to_csv(path, index=False)
            else:
                with path.open("wb") as handle:
                    pickle.dump(obj, handle)
            written.append(path)
            log.info("Wrote %s", path)

        write(f"formula_sample_support{tag}.json", self.sample_support)
        for (model, method), preds in self.cv_preds.items():
            write(f"cv_preds_{model}_{method}{tag}.pkl", preds)
            for year, bundle in self.coefs_by_year.get((model, method), {}).items():
                write(f"coefs_{model}_{method}{tag}_year{year}.pkl", bundle)
        for stem, weights in {**self.platt_weights, **self.mme_weights}.items():
            write(f"{stem}.pkl", weights)
        if self.cell_metrics is not None:
            write(f"summary_models{tag}.pkl", self.cell_metrics)
            write(f"summary_models{tag}.csv", self.cell_metrics)
        if self.pooled is not None:
            write(f"summary_models_pooled{tag}.csv", self.pooled)
        if self.yearly is not None:
            write(f"yearly_metrics_global{tag}.pkl", self.yearly)
            write(f"yearly_metrics_global{tag}.csv", self.yearly)
        return written


def evaluate(
    weekly: Weekly | pd.DataFrame,
    cv_spec: dict,
    dissemination: pd.DataFrame,
    *,
    workers: int = 1,
    results_dir: str | os.PathLike | None = None,
    budget_bytes: int | None = None,
    final_model: str | None = None,
) -> Evaluation:
    """Cross-validate the spec's formulas and score its baselines on `weekly`
    (a `features.Weekly`, or the legacy table).

    `dissemination` is the dissemination table as read from its CSV (see
    `prepare_inputs`): models train and are scored on its cells only, and
    predict every cell. Up to `workers` processes fit the global CV folds,
    as many as `budget_bytes` holds; results do not depend on either.
    `results_dir` is where the script read files from: relative
    `extras.external_predictions` files and, when MME weights are not
    optimised, the saved weights. `final_model` names a formula to also fit
    on every training year (`model.fit_final`), into `final_model` of the
    result: alongside the global CV folds, on their design matrix.

    Formula and CV errors raise; climatology, forecast and MME source
    problems warn and skip that model, as the script does.
    """
    if workers < 1:
        raise ValueError("workers must be a positive integer.")
    run = RunSettings.from_spec(cv_spec)
    cv_methods = coalesce(cv_spec.get("cv", {}).get("methods"), ["global"])
    weekly, cells, n_bins = prepare_inputs(weekly, cv_spec, dissemination)
    externals = _load_external_predictions(cv_spec.get("extras"), results_dir, n_bins)

    # Formulas come before sample support: only predictors that enabled
    # formulas reference may decide the common evaluation rows.
    formulas, resolution = build_formulas_from_spec(
        cv_spec, run.cutoff_mode, data=weekly, return_resolution=True, n_bins=n_bins
    )
    if final_model is not None and final_model not in formulas:
        raise ValueError(
            f"Model '{final_model}' not found in spec formulas. Available: {list(formulas)}"
        )
    rows, support = sample_support(weekly, formulas)
    support["formula_resolution"] = resolution
    log_sample_support(support)
    log.info("Formulas: %s", list(formulas))

    ctx = _Context(cv_spec, run, weekly, rows, cells, n_bins, budget_bytes=budget_bytes)
    result = Evaluation(output_tag=run.output_tag, sample_support=support, formulas=formulas)
    chosen_final_rows = None
    if final_model is not None:
        chosen_final_rows = final_rows(weekly, rows, cells, run, n_bins)
    scores = _Scores(cells, n_bins)
    _cross_validate(ctx, formulas, cv_methods, workers, result, scores, final_model)
    if final_model is not None and result.final_model is None:
        log.info("\nFitting %s on all %d training years...", final_model, len(run.fit_years))
        result.final_model = logit.fit_samples(formulas[final_model], weekly, chosen_final_rows)
    # After the CV, the scoring and the Platt calibration fan out over worker
    # processes (with one BLAS thread each); scores keep their order.
    with _scoring_pool(workers) as pool:
        scores.pool = ctx.pool = pool
        for external, preds in externals:
            log.info("\nExternal CV predictions: %s (%s)", external.name, external.path)
            scores.add_model(preds, external.name, external.method)
        clim_preds = _climatology_baselines(ctx, scores)
        calibrated_preds = _forecast_baselines(ctx, result, scores)
        _mme(ctx, clim_preds, calibrated_preds, workers, results_dir, result, scores)
        scores.resolve()

    if scores.cells:
        result.cell_metrics = pd.concat(scores.cells, ignore_index=True)
        try:
            result.pooled = summarize_models_pooled(result.cell_metrics)
        except Exception as e:
            warnings.warn(f"Could not compute pooled summary: {e}")
    if scores.yearly:
        result.yearly = pd.DataFrame(scores.yearly)
    return result


def fit_final(
    weekly: Weekly | pd.DataFrame,
    cv_spec: dict,
    dissemination: pd.DataFrame,
    model_name: str = "blended_model",
) -> logit.FittedModel:
    """Fit `model_name`'s formula on every training year that is not a true
    holdout, on the dissemination cells, after the evaluation's sample support.

    `weekly` is a `features.Weekly` or the legacy table; `dissemination` is
    the table as read from its CSV (see `prepare_inputs`). Training fits
    this alongside the CV instead (`evaluate(..., final_model=...)`).
    """
    run = RunSettings.from_spec(cv_spec)
    weekly, cells, n_bins = prepare_inputs(weekly, cv_spec, dissemination)
    formulas = build_formulas_from_spec(cv_spec, run.cutoff_mode, data=weekly, n_bins=n_bins)
    if model_name not in formulas:
        raise ValueError(
            f"Model '{model_name}' not found in spec formulas. Available: {list(formulas)}"
        )
    rows, support = sample_support(weekly, formulas)
    log_sample_support(support)
    train = final_rows(weekly, rows, cells, run, n_bins)
    log.info("\nFormula : %s", formulas[model_name])
    log.info("\nFitting %s on all %d training years...", model_name, len(run.fit_years))
    return logit.fit_samples(formulas[model_name], weekly, train)




def final_rows(weekly: Weekly, rows: np.ndarray, cells: pd.DataFrame, run: RunSettings, n_bins: int) -> np.ndarray:
    """The final fit's samples (positions): the supported `rows` in the
    dissemination cells and the fit years, with every outcome class."""
    fit_years = run.fit_years
    candidates = rows[in_cells(weekly, cells)[rows]]
    train = candidates[np.isin(weekly.take("year", candidates), fit_years)]
    if not len(train):
        raise ValueError(
            "Final fit has no rows in the configured training years and dissemination cells."
        )
    present = {value for value in pd.unique(weekly.take("outcome", train)) if isinstance(value, str)}
    missing = [value for value in interval_bins(n_bins) if value not in present]
    if missing:
        raise ValueError(
            "Final-fit training rows are missing outcome classes: " + ", ".join(missing)
        )
    log.info("  Training years        : %s", sorted(fit_years))
    log.info("  Training rows (allowed cells) : %d", len(train))
    return train


@dataclass
class _Context:
    spec: dict
    run: RunSettings
    weekly: Weekly  # every sample
    rows: np.ndarray  # the formula-supported samples, of every cell
    cells: pd.DataFrame  # dissemination cells
    n_bins: int
    budget_bytes: int | None = None
    pool: object = None  # the scoring pool, once the CV is done

    @functools.cached_property
    def year(self) -> np.ndarray:
        return self.weekly.take("year", self.rows)

    @functools.cached_property
    def allowed(self) -> np.ndarray:
        """Per supported sample: in a dissemination cell (models train on those)."""
        return in_cells(self.weekly, self.cells)[self.rows]

    def frame(self, where: np.ndarray | None = None, columns=None) -> pd.DataFrame:
        """The legacy table of the supported samples (where: a mask over
        them), with the given columns or all."""
        return self.weekly.frame(self.rows if where is None else self.rows[where], columns)

    def in_years(self, years) -> np.ndarray:
        return np.isin(self.year, list(years))

    def columns_like(self, prefix: str) -> list[str]:
        """The `<prefix>_week<n>` columns."""
        pattern = re.compile(rf"^{re.escape(prefix)}_week\d+$")
        return [c for c in self.weekly.columns if pattern.match(c)]


def _keys(weekly: Weekly, *extra) -> list[str]:
    wanted = set(PREDICTION_KEYS) | set(extra)
    return [c for c in weekly.columns if c in wanted]


def _slim(preds: pd.DataFrame, weekly: Weekly) -> pd.DataFrame:
    """CV predictions with only their keys beside the probabilities."""
    added = [c for c in preds.columns if c not in weekly.columns]
    return preds[[c for c in preds.columns if c in PREDICTION_KEYS] + added]


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


def _scored(preds: pd.DataFrame) -> pd.DataFrame:
    """The columns scoring reads, so a worker is sent no more (no dates, no inputs)."""
    return preds[[c for c in preds.columns if c in ("id", "year", "outcome") or c.startswith("cv_")]]


def _baseline_metrics(preds, allowed_cells, model, method):
    cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=allowed_cells)
    cell_metrics["model"] = model
    cell_metrics["cv_method"] = method
    return cell_metrics


@dataclass
class _Scores:
    """Per-cell metric frames and yearly metric rows, in the order models are scored."""

    allowed_cells: pd.DataFrame
    n_bins: int
    cells: list = field(default_factory=list)   # metric frames, or futures of them until resolve()
    yearly: list = field(default_factory=list)
    pool: object = None

    def add_model(self, preds, model, method):
        """CV predictions: per-cell and per-year metrics."""
        if self.pool is None:
            cell_metrics, yearly_rows = score_model_predictions(
                preds, model, method, self.allowed_cells, self.n_bins
            )
            self.cells.append(cell_metrics)
            self.yearly.extend(yearly_rows)
            return
        future = self.pool.submit(
            score_model_predictions, _scored(preds), model, method, self.allowed_cells, self.n_bins
        )
        self.cells.append((future, 0))
        self.yearly.append((future, 1))

    def add_baseline(self, preds, model, method):
        """Baselines and the MME: per-cell metrics only."""
        if self.pool is None:
            self.cells.append(_baseline_metrics(preds, self.allowed_cells, model, method))
        else:
            future = self.pool.submit(_baseline_metrics, _scored(preds), self.allowed_cells, model, method)
            self.cells.append((future, None))

    def resolve(self) -> None:
        """Wait for the pooled scores, keeping the order they were added in."""
        def value(entry):
            if not isinstance(entry, tuple):
                return [entry]
            future, part = entry
            result = future.result()
            return [result] if part is None else ([result[part]] if part == 0 else list(result[part]))
        self.cells = [frame for entry in self.cells for frame in value(entry)]
        yearly = []
        for entry in self.yearly:
            yearly.extend(value(entry) if isinstance(entry, tuple) else [entry])
        self.yearly = yearly


def _load_external_predictions(extras, results_dir, n_bins):
    """extras.external_predictions, loaded; relative files resolve against results_dir."""
    if results_dir is None:
        entries = (extras or {}).get("external_predictions") or []
        files = [(entry or {}).get("file") for entry in entries]
        relative = [file for file in files if file and not os.path.isabs(file)]
        if relative:
            raise ValueError(
                f"extras.external_predictions files {relative} are relative, "
                "but no results_dir was given to resolve them against."
            )
        results_dir = ""  # every file is absolute
    return [
        (external, load_external_predictions(external, n_bins))
        for external in parse_external_predictions(extras, os.fspath(results_dir))
    ]


def _cross_validate(ctx, formulas, methods, workers, result, scores, final_model=None):
    """Section 1: each formula under each CV method; with the global method,
    also the final fit of `final_model`."""
    run = ctx.run
    dtypes = ctx.weekly.column_dtypes()
    for model, formula in formulas.items():
        log.info("\nCV: %s", model)
        for method in methods:
            log.info("  Method: %s", method)
            try:
                coefs_by_year = {}
                plan = design.plan(formula, dtypes) if method == "global" else None
                if plan is not None:
                    preds, coefs_by_year, fitted = _global_cv(ctx, plan, workers, model == final_model)
                    if fitted is not None:
                        result.final_model = fitted
                elif method == "global":
                    # Models train on the dissemination cells only and predict every cell.
                    table = ctx.frame()
                    preds, coefs_by_year = compute_cv_global(
                        formula,
                        restrict_to_allowed(table, ctx.cells),
                        run.holdout_years,
                        training_years=run.training_years,
                        true_holdout_years=run.true_holdout_years,
                        data_pred=table,
                        n_jobs=workers,
                        save_coefs=True,
                        required_classes=interval_bins(ctx.n_bins),
                    )
                    preds = _slim(preds.reset_index(drop=True), ctx.weekly)
                elif method == "local":
                    preds = _slim(compute_cv_local(formula, ctx.frame(), run.holdout_years), ctx.weekly)
                elif method == "neighbors":
                    preds = _slim(compute_cv_neighbors(formula, ctx.frame(), run.holdout_years), ctx.weekly)
                elif method.startswith("cluster"):
                    cluster_var = ctx.spec.get("cv", {}).get("cluster_var", "cluster")
                    preds = compute_cv_clusters(formula, ctx.frame(), run.holdout_years, cluster_var)
                    preds = _slim(preds, ctx.weekly)
                else:
                    warnings.warn(f"Unknown CV method: {method}. Skipping.")
                    continue

                if preds is None or preds.empty:
                    raise ValueError(f"No predictions for {model}/{method}.")
                scores.add_model(preds, model, method)
                result.cv_preds[(model, method)] = preds
                if coefs_by_year:
                    result.coefs_by_year[(model, method)] = coefs_by_year
            except Exception as e:
                raise RuntimeError(f"Error in {model}/{method}: {e}") from e


def _global_cv(ctx, plan, workers, with_final):
    """compute_cv_global on the shared design: (predictions, coefficients by
    year, the final model or None)."""
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
        required_classes=interval_bins(ctx.n_bins),
        save_coefs=True,
        final_years=run.fit_years if with_final else None,
        workers=workers,
        budget_bytes=ctx.budget_bytes,
    )
    if not folds:
        raise ValueError("Global CV produced no fold predictions.")
    keys = _keys(ctx.weekly)
    preds = pd.concat(
        [pd.concat([ctx.weekly.frame(ctx.rows[test], keys), fold_preds], axis=1) for _, test, fold_preds, _ in folds],
        ignore_index=True,
    )
    coefs_by_year = {year: artifact for year, _, _, artifact in folds if artifact is not None}
    fitted = None
    if final is not None:
        clf, scaler, used, total = final
        logit.log_fit(used, total, clf.classes_, plan.columns)
        fitted = logit.FittedModel.from_estimators(plan.formula, plan.columns, clf, scaler)
    return preds, coefs_by_year, fitted


def _climatology_baselines(ctx, scores):
    """Section 2: extras.clim_logits. Returns the predictions by model name, for the MME."""
    preds_by_model = {}
    for task in (ctx.spec.get("extras") or {}).get("clim_logits") or []:
        name = task.get("name")
        if not name:
            raise ValueError("clim_logits task missing 'name'")
        prefix = task.get("base_col_prefix")
        if not prefix:
            raise ValueError("clim_logits task missing 'base_col_prefix'")
        starts = [int(y) for y in (task.get("window_start_years") or [])]
        end = task.get("window_end_year")
        # Only clim_raw has year windows; its 1900 window is the unwindowed column.
        if not starts or end is None or name != "clim_raw":
            variants = [(name, None, None)]
        else:
            variants = [(name if s == 1900 else f"{name}_{s}_{end}", s, end) for s in starts]

        holdout = ctx.in_years(ctx.run.holdout_years)
        for model, start_year, end_year in variants:
            try:
                if model == "unc_clim_raw":
                    earlier_col = task.get("earlier_col")
                    if not earlier_col:
                        raise ValueError(
                            "unc_clim_raw requires extras.clim_logits.earlier_col in YAML"
                        )
                    preds = make_raw_preds_from_wide_logit(
                        wide_df=ctx.frame(holdout, _keys(ctx.weekly, earlier_col, *ctx.columns_like(prefix))),
                        base_col_prefix=prefix,
                        holdout_years=ctx.run.holdout_years,
                        earlier_col=earlier_col,
                        earlier_is_logit=bool(task.get("earlier_is_logit", True)),
                        add_cv_earlier=True,
                        renormalize_6=True,
                    )
                else:
                    # The window suffix make_raw_preds_from_wide_logit_window reads.
                    windowed = start_year is not None and end_year is not None and start_year != 1900
                    window = f"_{start_year}_{end_year}" if windowed else ""
                    preds = make_raw_preds_from_wide_logit_window(
                        wide_df=ctx.frame(holdout, _keys(ctx.weekly, *ctx.columns_like(f"{prefix}{window}"))),
                        base_col_prefix=prefix,
                        holdout_years=ctx.run.holdout_years,
                        start_year=start_year,
                        end_year=end_year,
                    )
                if preds is None or preds.empty:
                    continue
                preds_by_model[model] = preds
                scores.add_baseline(preds, model, "raw")
            except Exception as e:
                warnings.warn(f"Error in clim logit '{model}': {e}")
    return preds_by_model


def _forecast_baselines(ctx, result, scores):
    """Section 3: extras.forecasts, raw and Platt-calibrated. Returns the
    calibrated predictions by MME key."""
    run = ctx.run
    calibrated_by_key = {}
    for fc_cfg in (ctx.spec.get("extras") or {}).get("forecasts") or []:
        fc_name = fc_cfg["name"]
        variant = coalesce(fc_cfg.get("variant"), "base")
        base_label = forecast_label(fc_name, variant)

        base = f"{fc_name}_p_onset{get_forecast_variant_suffix(ctx.spec, variant)}"
        if fc_cfg.get("raw", False):
            try:
                preds = make_raw_preds_from_wide(
                    ctx.frame(ctx.in_years(run.holdout_years), _keys(ctx.weekly, *ctx.columns_like(base))),
                    fc_name, variant, run.holdout_years, ctx.spec,
                )
                if preds is not None and not preds.empty:
                    scores.add_baseline(preds, f"{base_label}_raw", "raw")
            except Exception as e:
                warnings.warn(f"Error in raw forecast '{fc_name}': {e}")

        if fc_cfg.get("calibrated", False):
            try:
                preds = make_calibrated_preds_from_wide(
                    ctx.frame(
                        ctx.in_years([*run.training_years, *run.holdout_years]),
                        _keys(ctx.weekly, *ctx.columns_like(base)),
                    ),
                    fc_name,
                    variant,
                    run.training_years,
                    run.holdout_years,
                    run.true_holdout_years,
                    ctx.cells,
                    ctx.spec,
                    executor=ctx.pool,
                )
                if preds is not None and not preds.empty:
                    key = fc_name if variant == "base" else f"{fc_name}_calibrated_{variant}"
                    calibrated_by_key[key] = preds
                    model = f"{fc_name}_calibrated" if variant == "base" else key
                    scores.add_baseline(preds, model, "calibrated")
                    if fc_cfg.get("export_platt_weights", False):
                        weights = _platt_weights(ctx, fc_name, variant)
                        if weights is not None:
                            stem = f"platt_weights_{base_label}_calibrated_df{run.output_tag}"
                            result.platt_weights[stem] = weights
            except Exception as e:
                warnings.warn(f"Error in calibrated forecast '{fc_name}': {e}")
    return calibrated_by_key


def _platt_weights(ctx, fc_name, variant):
    """Platt weights fitted on the training years, for export (None without forecast columns)."""
    run = ctx.run
    base = f"{fc_name}_p_onset{get_forecast_variant_suffix(ctx.spec, variant)}"
    weeks = _present_weeks(ctx.weekly, base)
    if not weeks:
        return None
    frame = ctx.frame(ctx.in_years([*run.training_years, *run.holdout_years]), ["year", "outcome", *ctx.columns_like(base)])
    labels = [f"week{w}" for w in weeks]
    for w in weeks:
        frame[f"week{w}"] = frame[f"{base}_week{w}"]
    frame["later"] = np.maximum(0.0, 1.0 - frame[labels].sum(axis=1))
    return fit_platt_weights_export(frame, labels + ["later"], run.training_years, year_col="year")[
        "weights_df"
    ]


def _mme(ctx, clim_preds, calibrated_preds, workers, results_dir, result, scores):
    """Section 4: the MME blend of climatology and calibrated forecasts, with
    RPS-optimised weights, or saved ones when the holdouts are not 2000-2024."""
    mme_cfg = ctx.spec.get("mme") or {}
    if not (mme_cfg.get("enabled", False) and clim_preds):
        return
    run = ctx.run
    blend_models = mme_cfg.get("blend_models") or []
    names = [bm["name"] for bm in blend_models]
    weight_cols = [f"w_{name}" for name in names]
    bins = interval_bins(ctx.n_bins)
    cols = [f"cv_{b}" for b in bins]
    id_vars = ["time", "id", "lat", "lon", "year", "outcome"]
    optimise = bool(mme_cfg.get("optimize_if_holdout_is_full_2000_2024", True)) and sorted(
        run.holdout_years
    ) == list(range(2000, 2025))

    for variant in list(mme_cfg.get("variants") or ["fixed_cutoff"]):
        log.info("\n=== MME optimization (RPS-only) variant: %s ===", variant)
        sources = _mme_sources(blend_models, clim_preds, calibrated_preds)
        if sources is None:
            continue

        # Known defect, kept in phase 2: the sources have no lat/lon, so
        # merging two or more of them on id_vars raises KeyError 'lat'.
        joined = []
        for name in names:
            src = sources[name].copy()
            rename = {c: f"cv_{name}_{c[3:]}" for c in cols if c in src.columns}
            keep = [v for v in id_vars if v in src.columns] + [c for c in cols if c in src.columns]
            joined.append(src[keep].rename(columns=rename))
        base_full = joined[0]
        for other in joined[1:]:
            base_full = base_full.merge(other, on=id_vars, how="inner")
        base_allowed = restrict_to_allowed(base_full, ctx.cells)

        P_full = [base_full[[f"cv_{n}_{b}" for b in bins]].values.astype(float) for n in names]
        P_allowed = [
            clean_probs5(base_allowed[[f"cv_{n}_{b}" for b in bins]].values.astype(float))
            for n in names
        ]
        Y_allowed = np.column_stack([(base_allowed["outcome"] == b).astype(float) for b in bins])

        clim_model = next(
            (bm["name"] for bm in blend_models if bm.get("source") == "clim"), names[0]
        )
        stem = f"mme_optimized_weights{run.cutoff_tag}_{clim_model}_{variant}"
        if optimise:
            log.info("  Running MME optimization...")
            opt = optimize_mme_weights(P_allowed, Y_allowed, weight_cols, workers)
            weights, rps = opt["weights"], opt["rps"]
            result.mme_weights[f"{stem}{run.year_tag}"] = opt["weights_df"]
        else:
            log.info("  Reading saved MME weights...")
            read_tag = mme_cfg.get("weights_year_tag") or run.year_tag
            saved = _read_mme_weights(results_dir, f"{stem}{read_tag}.pkl")
            if saved is None:
                continue
            row = saved[saved["objective"] == "rps"].iloc[0]
            weights = np.array([row[w] for w in weight_cols])
            P_mix = sum(wi * Pi for wi, Pi in zip(weights, P_allowed))
            rps = pooled_rps5(clean_probs5(P_mix), Y_allowed)

        log.info("  MME weights: %s", dict(zip(names, np.round(weights, 4))))
        log.info("  RPS (pooled, allowed): %.6f", rps)
        P_mix_full = sum(wi * clean_probs5(Pi) for wi, Pi in zip(weights, P_full))
        P_mix_full = clean_probs5(P_mix_full)
        preds = base_full[[v for v in id_vars if v in base_full.columns]].copy()
        for j, b in enumerate(bins):
            preds[f"cv_{b}"] = P_mix_full[:, j]
        model = f"mme_{variant}_{clim_model}_opt_rps"
        scores.add_baseline(preds, model, "mme")
        log.info("  Added MME model: %s", model)


def _mme_sources(blend_models, clim_preds, calibrated_preds):
    """Each blend model's CV predictions by name; None, with a warning, if any is missing."""
    sources = {}
    for bm in blend_models:
        if bm.get("source") == "clim":
            src = clim_preds.get(bm["name"])
            if src is None:
                warnings.warn(f"MME blend_model '{bm['name']}' (source: clim) not found.")
                return None
        elif bm.get("source") == "forecast":
            cal_variant = bm.get("cal_variant", "base")
            key = bm["name"] if cal_variant == "base" else f"{bm['name']}_calibrated_{cal_variant}"
            src = calibrated_preds.get(key)
            if src is None:
                warnings.warn(f"MME blend_model '{bm['name']}' (key: {key}) not found.")
                return None
        else:
            warnings.warn(f"Unknown MME blend_model source: {bm.get('source')}")
            return None
        sources[bm["name"]] = src
    return sources


def _read_mme_weights(results_dir, name):
    """Saved MME weights from results_dir; None, with a warning, if absent."""
    if results_dir is None:
        warnings.warn(f"MME weights file not found: {name} (no results_dir given). Skipping.")
        return None
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        warnings.warn(f"MME weights file not found: {path}. Skipping.")
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)
