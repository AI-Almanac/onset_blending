"""The CRPS random forest blend: daily onset probabilities from each model's
daily rainfall and the climatology.

Its samples are the weekly samples (features.Weekly), and its inputs per
sample (features.DailyInputs) are the unit's lat and lon, each model's
ensemble-mean rainfall for every lead day, and the conditional climatology's
onset probability for lead days 1..max_day. It learns the observed onset's
lead day after the issue date (seasons without an onset count as a lead day
past max_day, or are left out), and predicts, per sample, the share of its
trees' leaf members on each lead day 1..max_day and later; the week bins are
their sums.

Everything is arrays. The numerics are onset-forest's (its fit, predict and
cross_validate, github.com/AI-Almanac/onset-forest at 61a7f5a): the same
feature columns in the same order, the same median imputation and targets,
and the same trees (_crps_tree), so a seeded forest reproduces its
predictions exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.features import DailyInputs, Weekly
from ._crps_tree import CRPSForest

__all__ = ["FittedForest", "Params", "cross_validate", "feature_matrix", "fit", "folds", "predict", "weekly_from_daily"]


@dataclass(frozen=True)
class Params:
    """How the forest blend is fitted (onset-forest's ForestParams, under the
    config's names, config.RandomForest) and cross-validated."""

    name: str = "blended_forest"
    trees: int = 100
    min_samples_leaf: int = 20
    max_features: str | None = "sqrt"
    max_samples: float = 1.0
    max_depth: int | None = None
    random_state: int = 123
    max_day: int = 28
    days_per_week: int = 7
    missing_onset: str = "after_week4"
    years_per_fold: int = 1
    n_jobs: int = -1

    @classmethod
    def from_config(cls, forest, days_per_week: int, n_jobs: int = -1) -> Params:
        return cls(
            name=forest.name,
            trees=forest.trees,
            min_samples_leaf=forest.min_samples_leaf,
            max_features=None if forest.max_features == "all" else forest.max_features,
            max_samples=forest.max_samples,
            max_depth=forest.max_depth,
            random_state=forest.random_state,
            max_day=forest.max_day,
            days_per_week=days_per_week,
            missing_onset=forest.missing_onset,
            years_per_fold=forest.years_per_fold,
            n_jobs=n_jobs,
        )


def feature_names(inputs: DailyInputs, model_names) -> list[str]:
    """lat, lon, each model's rain days ascending, then climatology days."""
    names = ["lat", "lon"]
    for model in model_names:
        days, _ = inputs.rain(model)
        names += [f"{model}_rain_mean_day_{int(d)}" for d in np.sort(days)]
    names += [f"clim_p_onset_day_{d}" for d in range(1, inputs.climatology.shape[1] + 1)]
    return names


def feature_matrix(weekly: Weekly, inputs: DailyInputs, rows: np.ndarray, model_names) -> np.ndarray:
    """The samples' features (rows, feature_names), float64, NaN where missing."""
    if not np.array_equal(inputs.data["time"].values, weekly["time"]):
        raise ValueError("the forest's inputs are not for these weekly samples")
    blocks = [weekly.take("lat", rows)[:, None], weekly.take("lon", rows)[:, None]]
    for model in model_names:
        days, rain = inputs.rain(model)
        blocks.append(rain[rows][:, np.argsort(days, kind="stable")])
    blocks.append(inputs.climatology[rows])
    return np.concatenate([np.asarray(b, dtype=np.float64) for b in blocks], axis=1)


def target(onset_lead: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """(samples the forest learns from, their lead day): onsets after the
    issue date; seasons without an onset at max_day + 1, or left out."""
    lead = np.asarray(onset_lead, dtype=np.float64).copy()
    if params.missing_onset == "after_week4":
        lead[np.isnan(lead)] = float(params.max_day + 1)
    elif params.missing_onset != "drop":
        raise ValueError("missing_onset must be 'after_week4' or 'drop'.")
    keep = ~np.isnan(lead) & (lead > 0)
    return keep, lead[keep]


@dataclass(frozen=True, eq=False)
class FittedForest:
    """A fitted forest: the features it reads (those observed in training),
    their training medians (imputing missing values), and the trees."""

    features: tuple[str, ...]
    columns: np.ndarray   # positions of `features` among the full feature names
    medians: np.ndarray
    forest: CRPSForest
    params: Params

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        meta, arrays = self.forest.to_arrays()
        params = {k: v for k, v in vars(self.params).items()}
        return (
            {"features": list(self.features), "forest": meta, "params": params},
            {**{f"tree_{k}": v for k, v in arrays.items()}, "columns": self.columns, "medians": self.medians},
        )

    @classmethod
    def from_arrays(cls, meta: dict, arrays) -> FittedForest:
        trees = {k[len("tree_"):]: arrays[k] for k in arrays if k.startswith("tree_")}
        return cls(
            features=tuple(meta["features"]),
            columns=np.asarray(arrays["columns"], dtype=np.int64),
            medians=np.asarray(arrays["medians"], dtype=np.float64),
            forest=CRPSForest.from_arrays(meta["forest"], trees),
            params=Params(**meta["params"]),
        )


# A tree-growing worker: its interpreter and compiled kernels, and per
# training row the node row sets, masks and targets of a tree's deepest path.
_WORKER_BASE_BYTES = 300 * 2**20
_WORKER_BYTES_PER_ROW = 96


def workers(train_rows: int, workers: int, budget_bytes: int | None) -> int:
    """Processes to grow trees in: up to `workers`, as many as the budget holds."""
    from ..core import parallel

    return parallel.workers_within([_WORKER_BASE_BYTES + train_rows * _WORKER_BYTES_PER_ROW], workers, budget_bytes)


def fit(X: np.ndarray, onset_lead: np.ndarray, names, params: Params) -> FittedForest:
    """onset_forest.fit on a feature matrix: the target's samples, the columns
    they observe, median-imputed."""
    keep, y = target(onset_lead, params)
    if not keep.any():
        raise ValueError("No training rows have an onset after their issue date.")
    rows = X[keep]
    observed = np.flatnonzero(~np.isnan(rows).all(axis=0))
    rows = rows[:, observed]
    medians = np.nanmedian(rows, axis=0)
    filled = np.where(np.isnan(rows), medians, rows)
    forest = CRPSForest(
        n_estimators=params.trees,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        max_samples=params.max_samples,
        max_depth=params.max_depth,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
    ).fit(filled, y)
    return FittedForest(tuple(names[i] for i in observed), observed, medians, forest, params)


def predict(fitted: FittedForest, X: np.ndarray) -> np.ndarray:
    """Daily onset probabilities (rows, max_day + 1): lead days 1..max_day, then later."""
    rows = X[:, fitted.columns]
    filled = np.where(np.isnan(rows), fitted.medians, rows)
    return fitted.forest.predict_lead_day_distribution(filled, max_day=fitted.params.max_day)


def weekly_from_daily(daily: np.ndarray, days_per_week: int) -> np.ndarray:
    """Week bins (rows, weeks + 1): week w sums lead days 7(w-1)+1..7w; later stays later."""
    n_days = daily.shape[1] - 1
    if n_days % days_per_week != 0:
        raise ValueError(f"{n_days} daily columns do not divide into {days_per_week}-day weeks.")
    weeks = daily[:, :n_days].reshape(len(daily), n_days // days_per_week, days_per_week).sum(axis=2)
    return np.column_stack([weeks, daily[:, n_days]])


def folds(training_years, cv_holdout_years, true_holdout_years=(), years_per_fold: int = 1):
    """(test years, train years) per fold: true holdouts first, one each, then
    the CV holdouts in contiguous blocks of up to years_per_fold; each trains
    on the training years minus its test years and every true holdout."""
    true_holdouts = [int(year) for year in true_holdout_years or ()]
    cv_years = [int(year) for year in cv_holdout_years]
    if years_per_fold > 1:
        cv_years = sorted(cv_years)
    blocks = [(year,) for year in true_holdouts] + [
        tuple(cv_years[start:start + years_per_fold]) for start in range(0, len(cv_years), years_per_fold)
    ]
    return [
        (block, tuple(sorted({int(y) for y in training_years if int(y) not in block and int(y) not in true_holdouts})))
        for block in blocks
    ]


def cross_validate(X: np.ndarray, onset_lead: np.ndarray, names, year: np.ndarray, trainable: np.ndarray,
                   scoreable: np.ndarray, fold_years, params: Params) -> list[tuple[np.ndarray, np.ndarray]]:
    """Out-of-fold daily probabilities: per fold, (positions of the scoreable
    samples in its test years, their probabilities), from a forest fitted on
    the trainable samples of its training years. Fold i is seeded
    random_state + i + 1; folds without test samples are skipped."""
    out = []
    for index, (test_years, train_years) in enumerate(fold_years):
        test = np.flatnonzero(scoreable & np.isin(year, test_years))
        if not len(test):
            continue
        train = np.flatnonzero(trainable & np.isin(year, train_years))
        fitted = fit(X[train], onset_lead[train], names, replace(params, random_state=params.random_state + index + 1))
        out.append((test, predict(fitted, X[test])))
    if not out:
        raise ValueError("No predict_on rows fall in any fold's test year.")
    return out
