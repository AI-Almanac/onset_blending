"""Probabilistic skill on arrays: Brier score, RPS, AUC and Pietra index of
week-bin probabilities, per unit, pooled and per year.

The numerics are the legacy scoring's (compute_cell_metrics_fast,
compute_yearly_metrics, summarize_models_pooled), operation for operation:

- Brier: the mean over samples of the squared distance across the interval
  bins (week1..weekN, later);
- RPS: the mean over samples of the squared distance between cumulative
  probabilities, over the 'earlier' bin too when a model has one;
- AUC: the Wilcoxon rank-sum statistic over every (sample, bin) pair;
- Pietra: the Kolmogorov-Smirnov statistic between the probabilities given
  to bins that happened and to bins that did not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata
from sklearn.metrics import roc_auc_score


def one_hot(outcome: np.ndarray, n_classes: int) -> np.ndarray:
    """(samples, classes) indicators of each sample's outcome code (-1: none)."""
    return np.column_stack([(outcome == k).astype(float) for k in range(n_classes)])


def brier(P: np.ndarray, Y: np.ndarray) -> float:
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def rps(P: np.ndarray, Y: np.ndarray) -> float:
    """Pooled RPS of probability and one-hot matrices over ordered bins."""
    CP = np.cumsum(P, axis=1)
    CY = np.cumsum(Y, axis=1)
    return float(np.mean(np.sum((CP - CY) ** 2, axis=1)))


def auc(y01: np.ndarray, score: np.ndarray) -> float:
    """AUC by the Wilcoxon rank-sum, over finite scores."""
    ok = np.isfinite(score) & ~np.isnan(y01)
    y = y01[ok].astype(int)
    s = score[ok]
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s, method="average")
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)


def pietra(P: np.ndarray, Y: np.ndarray) -> float:
    p = P.flatten()
    y = Y.flatten().astype(int)
    positive, negative = p[y == 1], p[y == 0]
    if len(positive) and len(negative):
        return float(ks_2samp(positive, negative).statistic)
    return np.nan


def clean(P: np.ndarray) -> np.ndarray:
    """Probabilities with non-finite or negative entries at 0, rows summing to
    1 (uniform where a row has nothing)."""
    P = np.array(P, dtype=float)
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= 0).flatten()
    if zero.any():
        P[zero] = 1.0 / P.shape[1]
        rs = P.sum(axis=1, keepdims=True)
    return P / rs


def cell_metrics(P: np.ndarray, outcome: np.ndarray, ids: np.ndarray, labels, *,
                 allowed: np.ndarray | None = None, earlier: np.ndarray | None = None) -> pd.DataFrame:
    """Per-unit Brier, RPS, AUC and n, and a pooled 'ALL' row that adds the
    Pietra index and each bin's Brier and AUC.

    P: (samples, bins) over the interval bins (`labels`); outcome: their codes
    (-1 without one); allowed: the samples scored (the dissemination units);
    earlier: the probability of an onset before the issue, prepended for the
    RPS. Samples without an outcome or with a missing probability are left
    out."""
    P = np.asarray(P, dtype=float)
    keep = (outcome >= 0) & ~np.isnan(P).any(axis=1)
    if allowed is not None:
        keep &= allowed
    if not keep.any():
        return pd.DataFrame()
    P, outcome, ids = P[keep], outcome[keep], np.asarray(ids)[keep]
    n_bins = P.shape[1]
    Y = one_hot(outcome, n_bins)
    if earlier is not None:
        early = np.asarray(earlier, dtype=float)[keep]
        P_rps = np.column_stack([np.where(np.isfinite(early), early, 0.0),
                                 np.where(np.isfinite(P), P, 0.0)])
        # No outcome is 'earlier': its indicator column is all zeros.
        Y_rps = np.column_stack([np.zeros(len(P)), Y])
    else:
        P_rps, Y_rps = P.copy(), Y.copy()

    units, unit = np.unique(ids.astype(str), return_inverse=True)
    order = np.argsort(unit, kind="stable")
    bounds = np.searchsorted(unit[order], np.arange(len(units) + 1))
    rows = []
    for k, name in enumerate(units):
        pos = order[bounds[k]:bounds[k + 1]]
        g_P, g_Y = P[pos], Y[pos]
        rows.append({
            "id": name,
            "brier": float(np.mean(np.sum((g_P - g_Y) ** 2, axis=1))),
            "rps": rps(P_rps[pos], Y_rps[pos]),
            "auc": auc(g_Y.flatten().astype(int), g_P.flatten()),
            "n": len(pos),
        })
    cells = pd.DataFrame(rows)

    pooled = {
        "id": "ALL",
        "lat": np.nan,
        "lon": np.nan,
        "brier": float(np.sum((P - Y) ** 2) / len(P)),
        "rps": rps(P_rps, Y_rps),
        "n": len(P),
        "auc": auc(Y.flatten().astype(int), P.flatten()),
        "pietra": pietra(P, Y),
    }
    for j, label in enumerate(labels):
        pooled[f"brier_{label}"] = float(np.mean((P[:, j] - Y[:, j]) ** 2))
    for j, label in enumerate(labels):
        pooled[f"auc_{label}"] = auc(Y[:, j].astype(int), P[:, j])
    return pd.concat([cells, pd.DataFrame([pooled])], ignore_index=True)


def yearly_metrics(P: np.ndarray, outcome: np.ndarray, year: np.ndarray, model: str, method: str) -> list[dict]:
    """Per-year pooled Brier, RPS and AUC over the interval bins, every unit."""
    P = np.asarray(P, dtype=float)
    keep = (outcome >= 0) & ~np.isnan(P).any(axis=1)
    P, outcome, year = P[keep], outcome[keep], np.asarray(year)[keep]
    Y = one_hot(outcome, P.shape[1])
    rows = []
    for y in np.unique(year):
        pos = np.flatnonzero(year == y)
        g_P, g_Y = P[pos], Y[pos]
        y_long, p_long = g_Y.flatten(), g_P.flatten()
        try:
            year_auc = roc_auc_score(y_long, p_long) if len(np.unique(y_long)) == 2 else np.nan
        except Exception:
            year_auc = np.nan
        rows.append({
            "year": int(y), "model": model, "cv_method": method,
            "brier": brier(g_P, g_Y), "rps": rps(g_P, g_Y), "auc": year_auc,
        })
    return rows


def pooled_skill(cells: pd.DataFrame, baseline: str = "unc_clim_raw") -> pd.DataFrame:
    """The pooled rows of every model, with skill against the baseline's."""
    pooled = cells[cells["id"] == "ALL"].copy()
    reference = pooled[pooled["model"] == baseline]
    if reference.empty:
        raise ValueError(f"Baseline model '{baseline}' not found in all_cells.")
    reference = reference.iloc[0]
    pooled["brier_skill"] = 1.0 - (pooled["brier"] / reference["brier"])
    pooled["rps_skill"] = 1.0 - (pooled["rps"] / reference["rps"])
    pooled["AUC diff"] = (pooled["auc"] - reference["auc"]) * 100.0
    return pooled
