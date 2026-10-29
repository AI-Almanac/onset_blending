"""The multinomial logistic blend: formula features, standardised, into a
multinomial logistic regression over the week bins.

`FittedModel` is a fitted blend (it converts to and from the coefficient
bundle the legacy scripts pickled, and to plain arrays and JSON-able
metadata; `fold_coefficients` stacks CV folds' models into one Dataset);
`fit_samples` fits one on weekly samples, from the arrays when the formula
has a design plan.

Cross-validation, on samples of the weekly features:

- `global_cv`: leave one year out, pooling every unit;
- `local_cv`: leave one year out within each unit;
- `cluster_cv`: leave one year out within each unit's cluster (one fit per
  cluster and year, predicting each of its units).

A formula without a design plan (functions, categorical terms) builds a
Patsy design per fit instead, from the fit's own training rows, as the
legacy CV did. Global CV and the final fit share one design matrix: a
formula's design for every evaluation sample is built once, column-major,
in shared memory that the fold workers and the final fit attach to. Each
fits its rows as the legacy compute_cv_global and fit_final did, bit for
bit, while holding one copy of its rows rather than two or three:

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
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
import pandas as pd
import patsy
import sklearn
import xarray as xr
from packaging.version import Version
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..core import design, parallel
from ..core.design import Plan
from ..core.features import Weekly

__all__ = [
    "FittedModel", "cluster_cv", "design_matrix", "fit", "fit_samples", "fold_coefficients", "fold_models",
    "global_cv", "local_cv", "multinomial",
]

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


def fold_coefficients(models: Mapping[int, FittedModel]) -> xr.Dataset:
    """CV folds' models, stacked over their holdout `year`: coef (year,
    class, feature), intercept (year, class) and the scaler statistics
    (year, feature). The folds share the formula, features and classes."""
    first = next(iter(models.values()))
    for year, model in models.items():
        if (model.formula, model.features, model.classes) != (first.formula, first.features, first.classes):
            raise ValueError(f"fold {year} has another formula, features or classes than the first fold")
        if model.scaler_mean is None:
            raise ValueError(f"fold {year} has no scaler")
    stack = lambda name: np.stack([getattr(m, name) for m in models.values()])
    return xr.Dataset(
        {
            "coef": (("year", "class", "feature"), stack("coef")),
            "intercept": (("year", "class"), stack("intercept")),
            **{name: (("year", "feature"), stack(name)) for name in _SCALER_ARRAYS},
            "n_samples": ("year", np.array([m.n_samples for m in models.values()], dtype=np.int64)),
        },
        coords={"year": np.array(list(models), dtype=np.int64), "class": list(first.classes),
                "feature": list(first.features)},
        attrs={"formula": first.formula},
    )


def fold_models(data: xr.Dataset) -> dict[int, FittedModel]:
    """The models fold_coefficients stacked, by holdout year."""
    features = [str(f) for f in data["feature"].values]
    classes = [str(c) for c in data["class"].values]
    out = {}
    for k, year in enumerate(data["year"].values.tolist()):
        out[int(year)] = FittedModel(
            formula=str(data.attrs["formula"]),
            features=features,
            classes=classes,
            coef=data["coef"].values[k],
            intercept=data["intercept"].values[k],
            n_samples=int(data["n_samples"].values[k]),
            **{name: data[name].values[k] for name in _SCALER_ARRAYS},
        )
    return out


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
    clf = multinomial()
    clf.fit(X_scaled, y_fit)
    return FittedModel.from_estimators(formula, features, clf, scaler)


def multinomial() -> LogisticRegression:
    """The blend's multinomial logistic regression (lbfgs, effectively
    unpenalised); sklearn before 1.5 has to be asked for multinomial."""
    kwargs = dict(solver="lbfgs", C=1e9, max_iter=5000, tol=1e-5)
    if Version(sklearn.__version__) < Version("1.5"):
        kwargs["multi_class"] = "multinomial"
    return LogisticRegression(**kwargs)


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
              required_classes=None, final_years=None, workers=1, budget_bytes=None):
    """The folds' predictions and models, and the final fit.

    values(name) gives a feature per sample; rows are the evaluation's
    samples (the design's rows), trainable marks those models train on (the
    dissemination cells), and year and outcome are per row. Returns
    ([(test_year, test_rows, probabilities, model)], final): probabilities
    (test rows, classes) in the model's class order, final being
    (classifier, scaler, rows used, rows) for the final_years fit, or None
    without final_years."""
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


def global_cv_frames(formula: str, weekly: Weekly, rows: np.ndarray, trainable: np.ndarray, *,
                     holdout_years, training_years, true_holdout_years=(), required_classes=None):
    """global_cv for a formula without a design plan: each fold builds its
    Patsy design from its own training rows. Returns the folds as global_cv does."""
    year = weekly.take("year", rows)
    outcome = weekly.take("outcome", rows)
    columns = _frame_columns(formula, weekly)
    out = []
    for test_year in holdout_years:
        train = np.flatnonzero(
            trainable & np.isin(year, training_years) & (year != test_year) & ~np.isin(year, true_holdout_years)
        )
        test = np.flatnonzero(year == test_year)
        _check_fold(test_year, train, test, outcome[train], training_years, required_classes)
        fitted = _fit_frames(formula, weekly.frame(rows[train], columns), outcome[train])
        if fitted is None:
            raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
        clf, scaler, info = fitted
        probabilities = _predict_frame(clf, scaler, info, weekly.frame(rows[test], columns))
        model = FittedModel.from_estimators(formula, info[1], clf, scaler)
        out.append((int(test_year), test, probabilities, model))
    return out


def _check_fold(test_year, train, test, outcome, training_years, required_classes):
    if not len(test):
        raise ValueError(f"Global CV holdout year {test_year} has no test rows.")
    if not len(train):
        raise ValueError(
            f"Global CV holdout year {test_year} has no training rows in "
            f"training_years={sorted(set(training_years))}."
        )
    present = {value for value in pd.unique(outcome) if isinstance(value, str)}
    if required_classes is not None:
        missing = [value for value in required_classes if value not in present]
        if missing:
            raise ValueError(
                f"Global CV holdout year {test_year} training rows are missing "
                f"outcome classes: {', '.join(missing)}."
            )
    if len(present) < 2:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")


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
    outcome = context["outcome"][train]
    _check_fold(test_year, train, test, outcome, context["training_years"], context["required_classes"])
    fitted = fit_fold(design, train, outcome, test)
    if fitted is None:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
    probabilities, clf, scaler = fitted
    return test_year, test, probabilities, FittedModel.from_estimators(
        context["formula_str"], context["columns"], clf, scaler
    )


def fit_fold(design: np.ndarray, train: np.ndarray, y: np.ndarray, test: np.ndarray):
    """fit_predict on rows of a column-major design: (probabilities for the
    test rows, classifier, scaler), or None."""
    n_columns = design.shape[1]
    stats, bad = [], np.array([not isinstance(v, str) for v in y])
    for j in range(n_columns):
        values = design[:, j][train]
        if np.isnan(values).any():
            # NaN-aware statistics count per column: the legacy route.
            return fit_predict(_rows(design, train), y, _rows(design, test))
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
    clf = _fit(X, y[~bad])
    return None if clf is None else (_predict(clf, X_test), clf, scaler)


def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray):
    """The legacy CV fit (_fit_predict_design): training rows standardised
    (in place) before the rows with a non-finite feature or no outcome are
    dropped, then the multinomial fit. Returns (probabilities for the test
    rows, NaN where one has a non-finite feature; the classifier; the
    scaler), or None when there is nothing to fit."""
    fitted = _fit_scaled(X_tr, y_tr)
    if fitted is None:
        return None
    clf, scaler = fitted
    return _predict(clf, scaler.transform(X_te)), clf, scaler


def _fit_scaled(X_tr: np.ndarray, y_tr: np.ndarray):
    """(classifier, scaler) fitted as fit_predict fits them, or None."""
    scaler = StandardScaler(copy=False)
    X_tr = scaler.fit_transform(X_tr)
    scaler.set_params(copy=True)  # as saved before: later transforms copy
    bad = ~np.isfinite(X_tr).all(axis=1) | np.array([not isinstance(v, str) for v in y_tr], dtype=bool)
    clf = _fit(X_tr[~bad], y_tr[~bad])
    return None if clf is None else (clf, scaler)


def _fit(X: np.ndarray, y: np.ndarray):
    """The multinomial fit on prepared rows, or None (no rows, one class, or
    a solver failure, which warns)."""
    if len(X) == 0 or len(np.unique(y)) < 2:
        return None
    clf = multinomial()
    try:
        clf.fit(X, y)
    except Exception as e:
        warnings.warn(f"_fit_predict_multinom fit failed: {e}")
        return None
    return clf


def _predict(clf, X: np.ndarray) -> np.ndarray:
    """Class probabilities, NaN for rows with a non-finite feature."""
    bad = ~np.isfinite(X).all(axis=1)
    probs = np.full((len(X), len(clf.classes_)), np.nan)
    if (~bad).any():
        probs[~bad] = clf.predict_proba(X[~bad])
    return probs


# -- local and cluster CV ------------------------------------------------------


def local_cv(formula: str, weekly: Weekly, rows: np.ndarray, holdout_years):
    """Each unit's own leave-one-year-out CV (compute_cv_local): for each unit
    and each holdout year it has, a fit on its other years' samples predicts
    that year's. rows are the evaluation's samples, in every unit and year.

    Returns [(test positions into rows, probabilities, classes)], by unit in
    order of first appearance, then holdout year; a fit with fewer than two
    outcome classes predicts nothing."""
    year = weekly.take("year", rows)
    outcome = weekly.take("outcome", rows)
    fits = _Fits(formula, weekly, rows, outcome)
    out = []
    for mine in _by_unit(weekly, rows):
        present = set(year[mine].tolist())
        for test_year in (y for y in holdout_years if y in present):
            train = mine[year[mine] != test_year]
            test = mine[year[mine] == test_year]
            if not len(train) or not len(test):
                continue
            predicted = fits.predict(fits.fit(train), test)
            if predicted is not None:
                out.append((test, *predicted))
    return out


def cluster_cv(formula: str, weekly: Weekly, rows: np.ndarray, holdout_years, cluster_var: str):
    """Leave-one-year-out CV within clusters (compute_cv_clusters): a unit's
    holdout year is predicted by the fit on the other years' samples of
    every unit in its cluster (its first sample's `cluster_var`). The fit is
    the same for every unit of a cluster, so it runs once per cluster and
    year. Returns as local_cv."""
    year = weekly.take("year", rows)
    outcome = weekly.take("outcome", rows)
    cluster = weekly.take(cluster_var, rows)
    fits = _Fits(formula, weekly, rows, outcome)
    cache = {}
    out = []
    for mine in _by_unit(weekly, rows):
        here = cluster[mine[0]]
        present = set(year[mine].tolist())
        for test_year in (y for y in holdout_years if y in present):
            key = (here, test_year)
            if key not in cache:
                train = np.flatnonzero((year != test_year) & (cluster == here))
                cache[key] = fits.fit(train) if len(train) else None
            test = mine[year[mine] == test_year]
            predicted = fits.predict(cache[key], test)
            if predicted is not None:
                out.append((test, *predicted))
    return out


def _by_unit(weekly: Weekly, rows: np.ndarray):
    """Positions into rows of each unit's samples, units in order of first appearance."""
    codes, units = pd.factorize(weekly.take("id", rows))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(units) + 1))
    return [order[bounds[k]:bounds[k + 1]] for k in range(len(units))]


class _Fits:
    """CV fits of one formula on positions into rows, from the arrays when it
    has a design plan and with Patsy designs otherwise, as
    _fit_predict_multinom fitted them."""

    def __init__(self, formula: str, weekly: Weekly, rows: np.ndarray, outcome: np.ndarray):
        self.formula, self.weekly, self.rows, self.outcome = formula, weekly, rows, outcome
        self.plan = design.plan(formula, weekly.column_dtypes())
        self.columns = None if self.plan is not None else _frame_columns(formula, weekly)

    def fit(self, train: np.ndarray):
        """(classifier, scaler, Patsy design or None), or None."""
        y = self.outcome[train]
        if len({v for v in pd.unique(y) if isinstance(v, str)}) < 2:
            return None
        if self.plan is not None:
            fitted = _fit_scaled(self.plan.build(self.weekly.take, self.rows[train], order="F"), y)
            return None if fitted is None else (*fitted, None)
        return _fit_frames(self.formula, self.weekly.frame(self.rows[train], self.columns), y)

    def predict(self, fitted, test: np.ndarray):
        """(probabilities, classes) for the test positions, or None without a fit."""
        if fitted is None:
            return None
        clf, scaler, info = fitted
        if info is not None:
            frame = self.weekly.frame(self.rows[test], self.columns)
            return _predict_frame(clf, scaler, info, frame), tuple(clf.classes_)
        X = scaler.transform(self.plan.build(self.weekly.take, self.rows[test], order="F"))
        return _predict(clf, X), tuple(clf.classes_)


def _frame_columns(formula: str, weekly: Weekly) -> list[str]:
    """The weekly columns a Patsy design of the formula may read."""
    return [c for c in dict.fromkeys(design.formula_columns(formula)) if c in weekly.columns]


def _fit_frames(formula: str, train: pd.DataFrame, y: np.ndarray):
    """fit_predict's fit on a Patsy design of the training table: (classifier,
    scaler, (design info, design columns)), or None."""
    matrix = patsy.dmatrix(design.right_hand_side(formula), train, return_type="dataframe")
    info = (matrix.design_info, [c for c in matrix.columns if c != "Intercept"])
    matrix = matrix.drop(columns=["Intercept"], errors="ignore")
    fitted = _fit_scaled(matrix.values.astype(float), y)
    return None if fitted is None else (*fitted, info)


def _predict_frame(clf, scaler, info, frame: pd.DataFrame) -> np.ndarray:
    """Probabilities for a test table, its design built with the training
    design's info, as _fit_predict_multinom predicted them."""
    design_info, columns = info
    matrix = patsy.dmatrix(design_info, frame, return_type="dataframe")
    matrix = matrix.drop(columns=["Intercept"], errors="ignore").reindex(columns=columns, fill_value=0.0)
    return _predict(clf, scaler.transform(matrix.values.astype(float)))


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
    clf = multinomial()
    clf.fit(X, y_fit)
    return clf, scaler, int(mask.sum()), len(y)
