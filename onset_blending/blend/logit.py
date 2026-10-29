"""The multinomial logistic blend: formula features, standardised, into a
multinomial logistic regression over the week bins.

`FittedModel` is a fitted blend (it converts to and from the coefficient
bundle the legacy scripts pickled, and to plain arrays and JSON-able
metadata); `fit_samples` fits one on weekly samples, from the arrays when the
formula has a design plan. Global leave-one-year-out cross-validation and the
final fit share one design matrix (`global_cv`): a formula's design for every
evaluation sample is built once, column-major, in shared memory that the
fold workers and the final fit attach to. Each fits its rows as the legacy
compute_cv_global and fit_final did, bit for bit, while holding one copy of
its rows rather than two or three:

- a fold computes its scaler statistics a column at a time (a design column
  is contiguous, so the sums run as on the fold's own column-major design)
  and writes its standardised rows straight into the row-major matrix the
  solver reads;
- the final fit gathers its rows row-major, as fit() did, and standardises
  them in place.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
import patsy
from sklearn.preprocessing import StandardScaler

from .._legacy.blend_evaluation_utils import _coef_artifact, _fit_and_predict, _make_multinom_clf
from ..core import design, parallel
from ..core.design import Plan
from ..core.features import Weekly

__all__ = ["FittedModel", "design_matrix", "fit", "fit_samples", "global_cv"]

log = logging.getLogger(__name__)

_SCALER_ARRAYS = ("scaler_mean", "scaler_scale", "scaler_var")


@dataclass(frozen=True, eq=False)
class FittedModel:
    """A multinomial logistic model on standardised formula features.

    Without a scaler (coefficient pickles from before the bundle format),
    the coefficients apply to the unscaled features.
    """

    formula: str  # the expanded formula that was fitted
    features: tuple[str, ...]  # design-matrix columns, in coefficient order
    classes: tuple[str, ...]  # outcome classes, in coefficient order
    coef: np.ndarray  # (n_classes, n_features)
    intercept: np.ndarray  # (n_classes,)
    scaler_mean: np.ndarray | None = None  # (n_features,)
    scaler_scale: np.ndarray | None = None
    scaler_var: np.ndarray | None = None  # kept so the legacy scaler is rebuilt exactly
    n_samples: int | None = None  # rows the scaler was fitted on

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "features", tuple(self.features))
        set_(self, "classes", tuple(self.classes))
        n_classes, n_features = len(self.classes), len(self.features)
        shapes = {"coef": (n_classes, n_features), "intercept": (n_classes,)}
        has_scaler = self.scaler_mean is not None
        if has_scaler:
            shapes.update({name: (n_features,) for name in _SCALER_ARRAYS})
        elif any(getattr(self, name) is not None for name in _SCALER_ARRAYS):
            raise ValueError("scaler_mean, scaler_scale and scaler_var go together.")
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value is None or np.shape(value) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {np.shape(value)}.")
            # C order, like apply_blend_model.py's coefficient matrix, so that
            # predict's matrix product runs the same BLAS kernel bit for bit.
            array = np.array(value, dtype=float, order="C")
            array.flags.writeable = False
            set_(self, name, array)
        if has_scaler != (self.n_samples is not None):
            raise ValueError("n_samples is required with, and only with, a scaler.")
        if has_scaler:
            set_(self, "n_samples", int(self.n_samples))

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Class probabilities `cv_<class>` for each row of `frame`, on its index.

        Rows with a non-finite feature get NaN. As apply_blend_model.py: the
        design is built from the saved formula and must have exactly the
        saved features, at least one of them finite.
        """
        features = list(self.features)
        design = design_matrix(self.formula, frame)
        missing = [f for f in features if f not in design]
        unexpected = [f for f in design if f not in features]
        if missing or unexpected:
            raise ValueError(
                "Applied design matrix does not match the saved training schema. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )
        design = design.loc[:, features]
        all_nonfinite = [
            column
            for column in features
            if not np.isfinite(
                pd.to_numeric(design[column], errors="coerce").to_numpy(dtype=float)
            ).any()
        ]
        if all_nonfinite:
            raise ValueError(
                "Saved model features have no finite values in the application "
                f"data: {all_nonfinite}."
            )
        if self.scaler_mean is not None:
            design = pd.DataFrame(
                self._scaler().transform(design.values.astype(float)),
                columns=features,
                index=design.index,
            )

        X = design[features].values.astype(float)
        scores = X @ self.coef.T + self.intercept
        bad = ~np.isfinite(X).all(axis=1)
        probs = np.full((len(X), len(self.classes)), np.nan)
        if (~bad).any():
            s = scores[~bad]
            s -= s.max(axis=1, keepdims=True)
            exp_s = np.exp(s)
            probs[~bad] = exp_s / exp_s.sum(axis=1, keepdims=True)
        return pd.DataFrame(probs, columns=[f"cv_{c}" for c in self.classes], index=design.index)

    @classmethod
    def from_estimators(cls, formula: str, features, clf, scaler) -> FittedModel:
        """From the fitted classifier and the scaler of its features."""
        return cls(
            formula=formula,
            features=features,
            classes=clf.classes_,
            coef=clf.coef_,
            intercept=clf.intercept_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            scaler_var=scaler.var_,
            n_samples=int(scaler.n_samples_seen_),
        )

    def to_legacy_bundle(self) -> dict:
        """The coefficient bundle 3_fit_final_model.py pickles: coefs, scaler,
        features, formula."""
        rows = [
            {
                "class": cls,
                "feature": feature,
                "coefficient": float(self.coef[i, j]),
                "intercept": float(self.intercept[i]),
            }
            for i, cls in enumerate(self.classes)
            for j, feature in enumerate(self.features)
        ]
        return {
            "coefs": pd.DataFrame(rows),
            "scaler": self._scaler() if self.scaler_mean is not None else None,
            "features": list(self.features),
            "formula": self.formula,
        }

    @classmethod
    def from_legacy_bundle(cls, bundle: dict, formula: str | None = None) -> FittedModel:
        """A model from a final or per-year coefficient bundle.

        `formula` is used only when the bundle has none (older bundles); as
        in apply_blend_model.py, classes are the sorted bundle classes and a
        feature missing from a class's coefficients counts as 0.
        """
        formula = bundle.get("formula") or formula
        if not formula:
            raise ValueError(
                "The coefficient bundle has no saved formula; pass the formula it was fitted with."
            )
        coef_df = bundle["coefs"]
        features = list(bundle["features"])
        classes = sorted(coef_df["class"].unique())
        coef = np.zeros((len(classes), len(features)))
        intercept = np.zeros(len(classes))
        for i, cls_ in enumerate(classes):
            cls_rows = coef_df[coef_df["class"] == cls_]
            intercept[i] = cls_rows["intercept"].iloc[0]
            for j, feature in enumerate(features):
                feature_row = cls_rows[cls_rows["feature"] == feature]
                if feature_row.empty:
                    warnings.warn(
                        f"Feature '{feature}' not found in coefs for class '{cls_}' — using 0"
                    )
                else:
                    coef[i, j] = float(feature_row["coefficient"].iloc[0])

        scaler = bundle.get("scaler")
        stats = {}
        if scaler is not None:
            if not (scaler.with_mean and scaler.with_std):
                raise ValueError("Only a centring and scaling StandardScaler is supported.")
            stats = dict(
                scaler_mean=scaler.mean_,
                scaler_scale=scaler.scale_,
                scaler_var=scaler.var_,
                n_samples=int(scaler.n_samples_seen_),
            )
        return cls(formula, features, classes, coef, intercept, **stats)

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """(JSON-able metadata, named float arrays): the model without pickles."""
        meta = {
            "formula": self.formula,
            "features": list(self.features),
            "classes": list(self.classes),
            "n_samples": self.n_samples,
        }
        names = ("coef", "intercept") + (_SCALER_ARRAYS if self.scaler_mean is not None else ())
        return meta, {name: getattr(self, name) for name in names}

    @classmethod
    def from_arrays(cls, meta: dict, arrays: dict[str, np.ndarray]) -> FittedModel:
        """The model `to_arrays` describes."""
        return cls(
            formula=meta["formula"],
            features=meta["features"],
            classes=meta["classes"],
            n_samples=meta["n_samples"],
            **{name: arrays.get(name) for name in ("coef", "intercept", *_SCALER_ARRAYS)},
        )

    def _scaler(self) -> StandardScaler:
        """The fitted StandardScaler these statistics came from."""
        # Fitting a placeholder gives sklearn's own attribute set, order and
        # types; the statistics are then swapped in.
        scaler = StandardScaler().fit(np.zeros((1, len(self.features))))
        scaler.n_samples_seen_ = type(scaler.n_samples_seen_)(self.n_samples)
        scaler.mean_ = self.scaler_mean.copy()
        scaler.var_ = self.scaler_var.copy()
        scaler.scale_ = self.scaler_scale.copy()
        return scaler


def design_matrix(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """The formula's right-hand side as a Patsy design matrix, without the
    intercept column and keeping rows with missing values."""
    rhs = formula.split("~", 1)[1].strip() if "~" in formula else formula
    return patsy.dmatrix(
        rhs, data, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[])
    ).drop(columns=["Intercept"], errors="ignore")


def fit(formula: str, train: pd.DataFrame) -> FittedModel:
    """Fit the formula on the rows with a known outcome and finite features."""
    design = design_matrix(formula, train)
    features = list(design.columns)
    y = train["outcome"].values
    valid = np.array([isinstance(v, str) for v in y])
    X = design.values.astype(float)
    mask = valid & np.isfinite(X).all(axis=1)
    X_fit, y_fit = X[mask], y[mask]
    if len(np.unique(y_fit)) < 2:
        raise ValueError("Fewer than 2 outcome classes in training data — cannot fit model.")
    log_fit(int(mask.sum()), len(train), np.unique(y_fit), features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)
    clf = _make_multinom_clf()
    clf.fit(X_scaled, y_fit)
    return FittedModel.from_estimators(formula, features, clf, scaler)


def log_fit(used: int, rows: int, classes, features) -> None:
    log.info("  Training rows used : %d / %d", used, rows)
    log.info("  Outcome classes    : %s", sorted(classes))
    log.info("  Feature columns    : %d", len(features))


def fit_samples(formula: str, weekly: Weekly, rows: np.ndarray) -> FittedModel:
    """`fit` on samples of the weekly features (positions), from the arrays
    when the formula has a design plan, with the same numerics."""
    plan = design.plan(formula, weekly.column_dtypes())
    if plan is None:
        return fit(formula, weekly.frame(rows))
    clf, scaler, used, total = fit_final_rows(
        lambda j: plan.column(j, weekly.take, rows), len(plan.columns), weekly.take("outcome", rows)
    )
    log_fit(used, total, clf.classes_, plan.columns)
    return FittedModel.from_estimators(formula, plan.columns, clf, scaler)


# ---------------------------------------------------------------------------
# Global cross-validation and the final fit, on one shared design
# ---------------------------------------------------------------------------

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
    with parallel.SharedArrays() as shared:
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
    count = min(parallel.workers_within([per_worker], workers, budget), n_tasks)
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
    parallel.limit_blas_threads(blas_threads)


def _run(task, context=None, design=None):
    context = _CONTEXT if context is None else context
    kind, years = task
    if design is None:  # in a worker
        design = parallel.attach({"design": context["design"]})["design"]
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
