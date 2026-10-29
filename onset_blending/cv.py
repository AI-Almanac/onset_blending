"""Global leave-one-year-out cross-validation, and the final fit, on one
design matrix.

A formula's design for every evaluation sample is built once from the
feature arrays (`design.Plan`), column-major, in shared memory that the
fold workers and the final fit attach to. Each fits its rows as
compute_cv_global and fit_final in the legacy code do, bit for bit, while
holding one copy of its rows rather than two or three:

- a fold computes its scaler statistics a column at a time (a design
  column is contiguous, so the sums run as on the fold's own column-major
  design) and writes its standardised rows straight into the row-major
  matrix the solver reads, rather than scaling a column-major copy that
  the solver copies again;
- the final fit gathers its rows row-major, as fit() did, and
  standardises them in place.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import _parallel
from ._legacy.blend_evaluation_utils import _coef_artifact, _fit_and_predict, _make_multinom_clf
from .design import Plan

log = logging.getLogger(__name__)

# A worker's interpreter and libraries, beyond the rows it fits.
_WORKER_BASE_BYTES = 350 * 2**20


def global_cv(plan: Plan, values, rows: np.ndarray, trainable: np.ndarray, year: np.ndarray,
              outcome: np.ndarray, *, holdout_years, training_years, true_holdout_years=(),
              required_classes=None, save_coefs=False, final_years=None, workers=1,
              budget_bytes=None):
    """The folds' predictions and coefficients, and the final fit.

    values(name) gives a feature per sample; rows are the evaluation's
    samples (the design's rows), trainable marks those models train on (the
    dissemination cells), and year and outcome are per row. Returns
    ([(test_year, test_rows, predictions, coefficient artifact)], final),
    final being (classifier, scaler, rows used, rows) for the final_years
    fit, or None without final_years."""
    tasks = [("fold", int(y)) for y in holdout_years]
    if final_years is not None:
        tasks.append(("final", tuple(int(y) for y in final_years)))
    context = {
        "formula_str": plan.formula,
        "columns": list(plan.columns),
        "year": np.asarray(year),
        "trainable": np.asarray(trainable, dtype=bool),
        "outcome": np.asarray(outcome, dtype=object),
        "training_years": tuple(training_years),
        "true_holdout_years": tuple(true_holdout_years),
        "required_classes": tuple(required_classes) if required_classes is not None else None,
        "save_coefs": bool(save_coefs),
    }
    shape = (len(rows), len(plan.columns))
    with _parallel.SharedArrays() as shared:
        design = plan.build(values, rows, out=shared.empty("design", shape, order="F"))
        context["design"] = shared.specs["design"]
        count = _workers(len(tasks), workers, shape, int(context["trainable"].sum()), budget_bytes)
        if count <= 1:
            results = [_run(task, context, design) for task in tasks]
        else:
            results = _run_pool(tasks, context, count)
        del design
    final = results.pop() if final_years is not None else None
    return results, final


def _workers(n_tasks: int, workers: int, shape, train_rows: int, budget_bytes) -> int:
    """Workers within the memory budget (after the shared design), then the
    fewest that keep the same number of rounds of tasks."""
    n_rows, n_columns = shape
    # The row-major fit matrix, the solver's per-row class arrays, the interpreter.
    per_worker = train_rows * 8 * (n_columns + 24) + _WORKER_BASE_BYTES
    budget = None if budget_bytes is None else max(budget_bytes - n_rows * n_columns * 8, 0)
    count = min(_parallel.workers_within([per_worker], workers, budget), n_tasks)
    if count <= 1:
        return 1
    rounds = math.ceil(n_tasks / count)
    return math.ceil(n_tasks / rounds)


def _run_pool(tasks, context, count):
    log.info("  Fitting %d global CV tasks with %d workers.", len(tasks), count)
    print(f"  Fitting {len(tasks)} global CV tasks with {count} workers.")
    blas_threads = max(1, (os.cpu_count() or 1) // count)
    with tempfile.TemporaryDirectory(prefix="global-cv-") as tmp:
        # Each worker loads the context from one file, rather than having it
        # pickled once per worker.
        path = os.path.join(tmp, "context.pkl")
        with open(path, "wb") as handle:
            pickle.dump(context, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with ProcessPoolExecutor(
            max_workers=count, mp_context=get_context("spawn"), initializer=_install, initargs=(path, blas_threads)
        ) as pool:
            return list(pool.map(_run, tasks))


_CONTEXT = None


def _install(path: str, blas_threads: int) -> None:
    global _CONTEXT
    with open(path, "rb") as handle:
        _CONTEXT = pickle.load(handle)
    _parallel.limit_blas_threads(blas_threads)


def _run(task, context=None, design=None):
    context = _CONTEXT if context is None else context
    kind, years = task
    if design is None:  # in a worker
        design = _parallel.attach({"design": context["design"]})["design"]
    if kind == "fold":
        return _fold(years, design, context)
    return _final(years, design, context)


def _fold(test_year: int, design: np.ndarray, context: dict):
    """One holdout year, as _compute_shared_fold fits it."""
    year, trainable = context["year"], context["trainable"]
    train = np.flatnonzero(
        trainable
        & np.isin(year, context["training_years"])
        & (year != test_year)
        & ~np.isin(year, context["true_holdout_years"])
    )
    test = np.flatnonzero(year == test_year)
    if not len(test):
        raise ValueError(f"Global CV holdout year {test_year} has no test rows.")
    if not len(train):
        raise ValueError(
            f"Global CV holdout year {test_year} has no training rows in "
            f"training_years={sorted(set(context['training_years']))}."
        )
    outcome = context["outcome"][train]
    present = {value for value in pd.unique(outcome) if isinstance(value, str)}
    if context["required_classes"] is not None:
        missing = [value for value in context["required_classes"] if value not in present]
        if missing:
            raise ValueError(
                f"Global CV holdout year {test_year} training rows are missing "
                f"outcome classes: {', '.join(missing)}."
            )
    if len(present) < 2:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
    preds, clf = fit_fold(design, train, outcome, test, context["columns"])
    if preds is None or clf is None:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
    return test_year, test, preds, _coef_artifact(test_year, clf, context)


def fit_fold(design: np.ndarray, train: np.ndarray, y: np.ndarray, test: np.ndarray, columns):
    """_fit_predict_design(scale=True) on rows of a column-major design:
    (probabilities for the test rows, classifier), or (None, None)."""
    n_columns = design.shape[1]
    stats, bad = [], np.array([not isinstance(v, str) for v in y])
    for j in range(n_columns):
        values = design[:, j][train]
        if np.isnan(values).any():
            # NaN-aware statistics count per column: the legacy route.
            return _fit_fold_copy(design, train, y, test, columns)
        stat = StandardScaler().fit(values[:, None])
        stats.append(stat)
        bad |= ~np.isfinite((values - stat.mean_[0]) / stat.scale_[0])
    scaler = _joined_scaler(stats, n_columns)
    keep = train[~bad]
    X = np.empty((len(keep), n_columns))
    for j in range(n_columns):
        X[:, j] = (design[:, j][keep] - scaler.mean_[j]) / scaler.scale_[j]
    X_test = scaler.transform(_rows(design, test))
    scaler.set_params(copy=True)  # as the legacy scaler is saved
    return _fit_and_predict(X, y[~bad], X_test, list(columns), pd.RangeIndex(len(test)), scaler, return_clf=True)


def _fit_fold_copy(design, train, y, test, columns):
    from ._legacy.blend_evaluation_utils import _fit_predict_design

    return _fit_predict_design(
        _rows(design, train), y, _rows(design, test), list(columns), pd.RangeIndex(len(test)),
        return_clf=True, scale=True,
    )


def _rows(design: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rows of the design, column-major as the fold's own design would be."""
    out = np.empty((len(rows), design.shape[1]), order="F")
    for j in range(design.shape[1]):
        out[:, j] = design[:, j][rows]
    return out


def _joined_scaler(stats: list[StandardScaler], n_columns: int) -> StandardScaler:
    """The scaler fitted on every column at once, from one per column."""
    # A placeholder fit sets sklearn's own attributes in their order; the
    # statistics are then swapped in.
    scaler = StandardScaler(copy=False).fit(np.zeros((1, n_columns)))
    scaler.n_samples_seen_ = stats[0].n_samples_seen_
    scaler.mean_ = np.concatenate([s.mean_ for s in stats])
    scaler.var_ = np.concatenate([s.var_ for s in stats])
    scaler.scale_ = np.concatenate([s.scale_ for s in stats])
    return scaler


def _final(years, design, context):
    rows = np.flatnonzero(context["trainable"] & np.isin(context["year"], years))
    return fit_final_rows(lambda j: design[:, j][rows], design.shape[1], context["outcome"][rows])


def fit_final_rows(column, n_columns: int, y: np.ndarray):
    """fit() in model.py, on design columns given per candidate row by
    column(j): the rows with an outcome and finite features, gathered
    row-major and standardised in place. Returns (classifier, scaler,
    rows used, rows)."""
    valid = np.array([isinstance(v, str) for v in y])
    finite = np.ones(len(y), dtype=bool)
    for j in range(n_columns):
        finite &= np.isfinite(column(j))
    mask = valid & finite
    y_fit = y[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Fewer than 2 outcome classes in training data — cannot fit model.")
    X = np.empty((int(mask.sum()), n_columns))
    for j in range(n_columns):
        X[:, j] = column(j)[mask]
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X)
    clf = _make_multinom_clf()
    clf.fit(X, y_fit)
    return clf, scaler, int(mask.sum()), len(y)
