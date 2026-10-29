"""Platt calibration of week-bin probabilities, on arrays.

`cross_validated` is the legacy platt_cv_multibin: for each holdout year,
each bin's probability is recalibrated by a logistic regression on the raw
probability (as R's glm(y ~ p) does, not on its logit), fitted on the other
years' samples outside the true holdouts and in the dissemination units; a
bin with too few such samples keeps its raw probability; rows are then
normalised. `export_weights` is fit_platt_weights_export: per-bin weights on
the logit, fitted on the training years, applied as expit(a + b logit(p)).

Fits run on one BLAS thread, so results do not depend on the machine's core
count; with an executor the holdout years are fitted in parallel worker
processes (inputs in shared memory), with the same results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from . import parallel

_EPS = 1e-6


def cross_validated(P: np.ndarray, outcome: np.ndarray, year: np.ndarray, trainable: np.ndarray,
                    holdout_years, *, executor=None) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Calibrated probabilities per holdout year: [(year, sample positions, (n, bins))].

    P: (samples, bins) raw probabilities; outcome: codes (-1 without one);
    trainable: samples a fit may train on (outside the true holdouts, in the
    dissemination units)."""
    P = np.asarray(P, dtype=float)
    n_bins = P.shape[1]
    known = outcome >= 0
    arrays = {
        "p": np.stack([np.clip(P[:, j], _EPS, 1 - _EPS) for j in range(n_bins)]),
        "y": np.stack([np.where(known, (outcome == j).astype(float), np.nan) for j in range(n_bins)]),
        "years": np.asarray(year),
        "trainable": np.asarray(trainable, dtype=bool),
    }
    if executor is None:
        folds = [_fold(arrays, test_year) for test_year in holdout_years]
    else:
        with parallel.SharedArrays(**arrays) as shared:
            futures = [executor.submit(_fold, shared.specs, test_year) for test_year in holdout_years]
            folds = [future.result() for future in futures]
    out = []
    for test_year, (calibrated, fallback) in zip(holdout_years, folds):
        test = np.flatnonzero(arrays["years"] == test_year)
        if not len(test):
            continue
        for j in fallback:
            calibrated[:, j] = P[test, j]
        sums = calibrated.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        out.append((int(test_year), test, calibrated / sums))
    return out


def _fold(arrays, test_year):
    """One holdout year's per-bin fits: (calibrated test probabilities, bins
    left uncalibrated). arrays holds p, y (bin, sample), years and trainable,
    or their shared-memory specs."""
    from threadpoolctl import threadpool_limits

    if not isinstance(arrays["p"], np.ndarray):
        specs = arrays
        try:
            return _fold(parallel.attach(specs), test_year)
        finally:
            parallel.detach(specs)
    p_all, y_all, years = arrays["p"], arrays["y"], arrays["years"]
    test = years == test_year
    train = np.flatnonzero((years != test_year) & arrays["trainable"])
    calibrated = np.full((int(test.sum()), len(p_all)), np.nan)
    fallback = []
    with threadpool_limits(limits=1):
        for j in range(len(p_all)):
            p_tr, y_tr = p_all[j][train], y_all[j][train]
            ok = np.isfinite(p_tr) & ~np.isnan(y_tr)
            if ok.sum() < 10 or len(np.unique(y_tr[ok])) < 2:
                fallback.append(j)
                continue
            clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
            clf.fit(p_tr[ok].reshape(-1, 1), y_tr[ok])
            calibrated[:, j] = expit(clf.intercept_[0] + clf.coef_[0][0] * p_all[j][test])
    return calibrated, fallback


def export_weights(P: np.ndarray, outcome: np.ndarray, labels) -> pd.DataFrame:
    """Per-bin Platt weights on the logit (bin, intercept, slope), fitted on
    the samples given (the training years); (0, 1) where a bin cannot be."""
    rows = []
    known = outcome >= 0
    for j, label in enumerate(labels):
        p = np.clip(np.asarray(P[:, j], dtype=float), _EPS, 1 - _EPS)
        y = np.where(known, (outcome == j).astype(float), np.nan)
        ok = np.isfinite(p) & ~np.isnan(y)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            rows.append({"bin": label, "intercept": 0.0, "slope": 1.0})
            continue
        clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
        clf.fit(logit(p[ok]).reshape(-1, 1), y[ok])
        rows.append({"bin": label, "intercept": float(clf.intercept_[0]), "slope": float(clf.coef_[0][0])})
    return pd.DataFrame(rows)
