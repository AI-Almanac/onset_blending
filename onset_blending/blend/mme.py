"""The multi-model ensemble: a weighted mixture of other predictions (the
climatology baselines and calibrated forecasts), weights chosen to minimise
the pooled RPS in the dissemination units.

The weights start from the best point of a coarse simplex grid (step 0.1)
and are refined by BFGS in softmax space, as the legacy optimize_mme_weights
(and R's Section 4) did. Sources are aligned on their samples' keys (id and
issue date), so any number of them mix.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import minimize

from ..core import metrics
from ..core.predictions import Predictions

__all__ = ["align", "mix", "optimise_weights"]


def align(sources: list[Predictions]) -> tuple[list[np.ndarray], np.ndarray]:
    """Each source's probabilities on the samples every source has, in the
    first source's order: ([(n, bins)], positions into the first source)."""
    def keys(p: Predictions):
        times = p.data["time"].values if "time" in p.data.coords else np.zeros(len(p))
        return list(zip(p.ids.tolist(), np.asarray(times).astype("datetime64[D]").tolist()))

    first = keys(sources[0])
    positions = []
    for source in sources[1:]:
        index = {key: i for i, key in enumerate(keys(source))}
        positions.append(index)
    common = [i for i, key in enumerate(first) if all(key in index for index in positions)]
    common = np.asarray(common, dtype=np.int64)
    aligned = [sources[0].probabilities[common]]
    for source, index in zip(sources[1:], positions):
        aligned.append(source.probabilities[[index[first[i]] for i in common]])
    return aligned, common


def mix(weights: np.ndarray, probabilities: list[np.ndarray]) -> np.ndarray:
    """The weighted mixture of cleaned probabilities, cleaned."""
    return metrics.clean(sum(w * metrics.clean(P) for w, P in zip(weights, probabilities)))


def mixed_rps(weights: np.ndarray, cleaned: list[np.ndarray], Y: np.ndarray) -> float:
    return metrics.rps(metrics.clean(sum(w * P for w, P in zip(weights, cleaned))), Y)


def optimise_weights(cleaned: list[np.ndarray], Y: np.ndarray) -> tuple[np.ndarray, float]:
    """Weights minimising the pooled RPS of the mixture, and that RPS."""
    best_rps, best = np.inf, None
    for w in _simplex_grid(len(cleaned), step=0.1):
        value = mixed_rps(w, cleaned, Y)
        if np.isfinite(value) and value < best_rps:
            best_rps, best = value, w
    if best is None:
        best = _simplex_grid(len(cleaned), step=0.1)[0]
    w0 = np.maximum(best, 1e-6)
    w0 = w0 / w0.sum()
    result = minimize(
        fun=lambda theta: mixed_rps(_softmax(theta), cleaned, Y), x0=np.log(w0), method="BFGS",
        options={"maxiter": 200},
    )
    weights = _softmax(result.x)
    return weights, mixed_rps(weights, cleaned, Y)


def _simplex_grid(n: int, step: float = 0.1) -> np.ndarray:
    if n == 1:
        return np.array([[1.0]])
    values = np.arange(0, 1 + step / 2, step)
    grid = []
    for combo in product(*([values] * (n - 1))):
        total = sum(combo)
        if total <= 1.0 + 1e-9:
            grid.append(list(combo) + [1.0 - total])
    return np.array(grid)


def _softmax(theta: np.ndarray) -> np.ndarray:
    e = np.exp(theta - np.max(theta))
    return e / e.sum()
