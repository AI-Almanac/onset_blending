"""The CRPS random forest over onset lead days: the trees and their split search.

From onset-forest (github.com/AI-Almanac/onset-forest, forest.py at 61a7f5a),
itself ported from the science team's notebook
(crps_rf_three_experiments_engineered_raw_with_without_climatology.ipynb). The
split search, node-level feature sampling and RNG consumption order are
unchanged, so a seeded fit grows onset-forest's (and the notebook's) trees
exactly. Added: a fitted forest as flat arrays (`to_arrays`, `from_arrays`), so
a trained blend stores it without pickles.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from joblib import Parallel, delayed
from numba import njit


def _jit(**options):
    """njit with on-disk caching when a cache location is writable.

    Numba resolves a cache locator at decoration time (the package's __pycache__, then
    NUMBA_CACHE_DIR or the user cache dir) and raises if none is writable, e.g. a
    read-only install inside a container. Fall back to in-memory compilation then.
    """

    def decorate(fn):
        try:
            return njit(cache=True, nogil=True, **options)(fn)
        except RuntimeError:
            return njit(cache=False, nogil=True, **options)(fn)

    return decorate


@_jit(inline="always")
def _fen_add(bit, i, delta):
    n = bit.shape[0] - 1
    while i <= n:
        bit[i] += delta
        i += i & (-i)


@_jit(inline="always")
def _fen_sum(bit, i):
    out = 0.0
    while i > 0:
        out += bit[i]
        i -= i & (-i)
    return out


@_jit(inline="always")
def _dist_to_set(bit_count, bit_sum, total_count, total_sum, idx, val):
    count_le = _fen_sum(bit_count, idx)
    sum_le = _fen_sum(bit_sum, idx)
    count_gt = total_count - count_le
    sum_gt = total_sum - sum_le
    return val * count_le - sum_le + sum_gt - val * count_gt


@_jit()
def _scan_feature(x_sorted, y_sorted, yi_sorted, m, n, parent_T, total_sum, min_leaf):
    """One feature scan. Returns (best_score, best_threshold, found)."""
    lc = np.zeros(m + 1)
    ls = np.zeros(m + 1)
    rc = np.zeros(m + 1)
    rs = np.zeros(m + 1)

    for k in range(n):
        _fen_add(rc, yi_sorted[k], 1.0)
        _fen_add(rs, yi_sorted[k], y_sorted[k])

    n_left = 0
    n_right = n
    total_left = 0.0
    total_right = total_sum
    T_left = 0.0
    T_right = parent_T

    best_score = np.inf
    best_threshold = np.nan
    found = False

    pos = 0
    while pos < n:
        x_val = x_sorted[pos]
        next_pos = pos
        while next_pos < n and x_sorted[next_pos] == x_val:
            val = y_sorted[next_pos]
            idx = yi_sorted[next_pos]

            T_right -= _dist_to_set(rc, rs, n_right, total_right, idx, val)
            _fen_add(rc, idx, -1.0)
            _fen_add(rs, idx, -val)
            n_right -= 1
            total_right -= val

            T_left += _dist_to_set(lc, ls, n_left, total_left, idx, val)
            _fen_add(lc, idx, 1.0)
            _fen_add(ls, idx, val)
            n_left += 1
            total_left += val

            next_pos += 1

        if next_pos < n and n_left >= min_leaf and n_right >= min_leaf:
            score = T_left / n_left + T_right / n_right
            if score < best_score:
                next_x = x_sorted[next_pos]
                best_threshold = x_val + 0.5 * (next_x - x_val)
                best_score = score
                found = True

        pos = next_pos

    return best_score, best_threshold, found


@_jit()
def _best_split_kernel(X, rows, y, yi, feature_ids, parent_T, total_sum, m, min_leaf):
    """Best split of X's `rows` over the candidate features; y and yi are those rows'
    targets. Returns (feature, threshold, score, found)."""
    n = rows.shape[0]
    best_score = np.inf
    best_feature = -1
    best_threshold = np.nan
    found = False

    for f in range(feature_ids.shape[0]):
        feature = feature_ids[f]
        col = X[rows, feature]
        order = np.argsort(col, kind="mergesort")
        x_sorted = col[order]
        if x_sorted[0] == x_sorted[-1]:
            continue
        y_sorted = y[order]
        yi_sorted = yi[order]

        sc, thr, fnd = _scan_feature(
            x_sorted, y_sorted, yi_sorted, m, n, parent_T, total_sum, min_leaf
        )
        if fnd and sc < best_score:
            best_score = sc
            best_feature = feature
            best_threshold = thr
            found = True

    return best_feature, best_threshold, best_score, found


@_jit()
def _pairwise_abs_sum(y_sorted):
    """T = sum_{i<j}|y_i - y_j| for ascending y_sorted."""
    total = 0.0
    csum = 0.0
    for i in range(y_sorted.shape[0]):
        total += y_sorted[i] * i - csum
        csum += y_sorted[i]
    return total


def _compress_y(y):
    """Return (yi_1based, m) where yi are dense ranks of y."""
    uniq, inv = np.unique(y, return_inverse=True)
    return (inv + 1).astype(np.int64), int(uniq.shape[0])


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "values")

    def __init__(self, feature=None, threshold=None, left=None, right=None, values=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.values = values

    @property
    def is_leaf(self):
        return self.values is not None

    def leaf_values(self, x):
        node = self
        while node.values is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.values


class _CRPSTree:
    def __init__(self, min_samples_leaf=5, random_state=None, max_depth=None, max_features="sqrt"):
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = random_state
        self.max_depth = max_depth
        self.max_features = max_features
        self.root_ = None
        self.n_features_in_ = None
        self.rng_ = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        return self.fit_rows(X, y, np.arange(X.shape[0], dtype=np.int64))

    def fit_rows(self, X, y, rows):
        """Grow the tree on X[rows] and y[rows], in that order, without copying X.

        Nodes hold row indices into X and the split kernel reads each candidate
        column through them, so a node keeps O(rows) memory rather than a copy
        of its rows of X. The tree is the one fit(X[rows], y[rows]) grows."""
        self.n_features_in_ = X.shape[1]
        self.root_ = self._grow(X, y, np.asarray(rows, dtype=np.int64), 0)
        return self

    def _grow(self, X, y_all, rows, depth):
        n = rows.shape[0]
        p = X.shape[1]
        y = y_all[rows]
        ml = self.min_samples_leaf
        if n < 2 * ml:
            return _Node(values=y)
        if self.max_depth is not None and depth >= self.max_depth:
            return _Node(values=y)
        if np.all(y == y[0]):
            return _Node(values=y)

        y_sorted = np.sort(y)
        parent_T = _pairwise_abs_sum(y_sorted)
        parent_score = parent_T / n
        total_sum = float(y_sorted.sum())

        yi, m = _compress_y(y)
        feature_ids = self._candidate_features(p)

        feat, thr, score, found = _best_split_kernel(
            X, rows, y, yi, feature_ids, parent_T, total_sum, m, ml
        )

        if not found:
            return _Node(values=y)
        if score >= parent_score - 1e-12:
            return _Node(values=y)

        left_mask = X[rows, feat] <= thr
        nl = left_mask.sum()
        if nl < ml or (n - nl) < ml:
            return _Node(values=y)

        left = self._grow(X, y_all, rows[left_mask], depth + 1)
        right = self._grow(X, y_all, rows[~left_mask], depth + 1)
        return _Node(feature=int(feat), threshold=float(thr), left=left, right=right)

    def _candidate_features(self, p):
        """Sample candidate predictors independently at each node, as sklearn RF does."""
        if self.max_features is None:
            return np.arange(p, dtype=np.int64)
        if self.max_features == "sqrt":
            n_candidates = max(1, int(np.sqrt(p)))
            return self.rng_.choice(p, size=n_candidates, replace=False).astype(np.int64)
        raise ValueError('max_features must be "sqrt" or None.')

    def predict_members_one(self, x):
        return self.root_.leaf_values(x)


def _fit_one_tree(X, y, idx, min_samples_leaf, max_depth, max_features, seed):
    tree = _CRPSTree(
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        max_depth=max_depth,
        max_features=max_features,
    )
    tree.fit_rows(X, y, idx)
    return tree


def _route_to_leaves(
    node: _Node, X: np.ndarray, rows: np.ndarray
) -> Iterator[tuple[_Node, np.ndarray]]:
    """Yield (leaf, row indices) for every leaf reached by some row of X."""
    stack = [(node, rows)]
    while stack:
        node, rows = stack.pop()
        if rows.size == 0:
            continue
        if node.is_leaf:
            yield node, rows
            continue
        goes_left = X[rows, node.feature] <= node.threshold
        stack.append((node.left, rows[goes_left]))
        stack.append((node.right, rows[~goes_left]))


def lead_day_histogram(members: np.ndarray, max_day: int) -> np.ndarray:
    """Counts of members equal to each lead day 1..max_day, then a count of members > max_day."""
    members = np.asarray(members, dtype=np.float64)
    days = np.arange(1, max_day + 1, dtype=np.float64)
    per_day = (members[:, None] == days[None, :]).sum(axis=0)
    later = np.count_nonzero(members > max_day)
    return np.append(per_day, later).astype(np.float64)


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    return counts / counts.sum(axis=1, keepdims=True)


class CRPSForest:
    """Bootstrapped CRPS trees; predictions pool the leaf members of every tree."""

    def __init__(
        self,
        n_estimators=100,
        min_samples_leaf=5,
        bootstrap=True,
        max_samples=None,
        random_state=None,
        max_depth=None,
        n_jobs=-1,
        max_features="sqrt",
    ):
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.bootstrap = bool(bootstrap)
        self.max_samples = max_samples
        self.random_state = random_state
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.max_features = max_features

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._check_fit_inputs(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_samples_bootstrap_ = self._resolve_bootstrap_size(X.shape[0])

        seeds, idxs = self._draw_tree_seeds_and_samples(X.shape[0])
        self.trees_ = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_one_tree)(
                X, y, idxs[i], self.min_samples_leaf, self.max_depth, self.max_features, seeds[i]
            )
            for i in range(self.n_estimators)
        )
        self.feature_importances_ = self._split_count_importances()
        return self

    def _check_fit_inputs(self, X, y):
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have incompatible lengths.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1.")
        if self.max_features not in {"sqrt", None}:
            raise ValueError('max_features must be "sqrt" or None.')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("This implementation requires finite numeric X and y.")

    def _resolve_bootstrap_size(self, n_samples):
        if self.bootstrap:
            return self._resolve_max_samples(n_samples)
        if self.max_samples is not None:
            raise ValueError("max_samples cannot be set if bootstrap=False.")
        return n_samples

    def _draw_tree_seeds_and_samples(self, n_samples):
        """For each tree draw its seed first, then its bootstrap indices (reference order)."""
        rng = np.random.default_rng(self.random_state)
        base = np.arange(n_samples)
        seeds = []
        idxs = []
        for _ in range(self.n_estimators):
            seeds.append(int(rng.integers(0, np.iinfo(np.int32).max)))
            if self.bootstrap:
                idxs.append(rng.integers(0, n_samples, size=self.n_samples_bootstrap_))
            else:
                idxs.append(base)
        return seeds, idxs

    def _split_count_importances(self):
        """Fraction of all internal-node splits assigned to each predictor."""
        split_counts = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.trees_:
            stack = [tree.root_]
            while stack:
                node = stack.pop()
                if node is None or node.is_leaf:
                    continue
                split_counts[node.feature] += 1.0
                stack.append(node.left)
                stack.append(node.right)
        total_splits = split_counts.sum()
        return split_counts / total_splits if total_splits > 0 else split_counts

    def _resolve_max_samples(self, n_samples):
        ms = self.max_samples
        if ms is None:
            return n_samples
        if isinstance(ms, (bool, np.bool_)):
            raise ValueError("max_samples must be None, int in [1,n], or float in (0,1].")
        if isinstance(ms, (int, np.integer)):
            ms = int(ms)
            if not 1 <= ms <= n_samples:
                raise ValueError("max_samples int must be in [1, n_samples].")
            return ms
        if isinstance(ms, (float, np.floating)):
            ms = float(ms)
            if not 0.0 < ms <= 1.0:
                raise ValueError("max_samples float must be in (0.0, 1.0].")
            return max(1, int(round(n_samples * ms)))
        raise ValueError("max_samples must be None, int in [1,n], or float in (0,1].")

    def _as_feature_matrix(self, X):
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return X

    def predict_members(self, X):
        X = self._as_feature_matrix(X)
        return [np.concatenate([t.predict_members_one(x) for t in self.trees_]) for x in X]

    def predict(self, X):
        return np.array([v.mean() for v in self.predict_members(X)], dtype=float)

    def predict_quantiles(self, X, quantiles=(0.1, 0.5, 0.9)):
        q = np.asarray(quantiles, dtype=float)
        return np.vstack([np.quantile(v, q) for v in self.predict_members(X)])

    def predict_lead_day_distribution(self, X, max_day=28):
        """(n, max_day + 1) shares of pooled leaf members per lead day 1..max_day, then > max_day.

        Equivalent to histogramming predict_members row by row, but routes all rows
        through each tree at once.
        """
        X = self._as_feature_matrix(X)
        counts = np.zeros((X.shape[0], max_day + 1), dtype=np.float64)
        all_rows = np.arange(X.shape[0])
        for tree in self.trees_:
            for leaf, rows in _route_to_leaves(tree.root_, X, all_rows):
                counts[rows] += lead_day_histogram(leaf.values, max_day)
        return _normalize_rows(counts)

    def _check_is_fitted(self):
        if not hasattr(self, "trees_"):
            raise RuntimeError("Estimator is not fitted yet. Call fit(X, y) first.")

    # -- as arrays ----------------------------------------------------------

    _PARAMS = ("n_estimators", "min_samples_leaf", "bootstrap", "max_samples", "random_state",
               "max_depth", "n_jobs", "max_features")

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """The fitted trees as flat arrays: every node of every tree in preorder,
        with its split (feature, threshold) and children, or its leaf members."""
        self._check_is_fitted()
        feature, threshold, left, right, start, count, values, roots = [], [], [], [], [], [], [], []
        n_values = 0
        for tree in self.trees_:
            roots.append(len(feature))
            stack = [(tree.root_, None, None)]  # (node, parent index, is left child)
            while stack:
                node, parent, is_left = stack.pop()
                index = len(feature)
                if parent is not None:
                    (left if is_left else right)[parent] = index
                if node.is_leaf:
                    feature.append(-1)
                    threshold.append(np.nan)
                    left.append(-1)
                    right.append(-1)
                    start.append(n_values)
                    count.append(len(node.values))
                    values.append(np.asarray(node.values, dtype=np.float64))
                    n_values += len(node.values)
                else:
                    feature.append(node.feature)
                    threshold.append(node.threshold)
                    left.append(-1)
                    right.append(-1)
                    start.append(-1)
                    count.append(0)
                    stack.append((node.right, index, False))
                    stack.append((node.left, index, True))
        arrays = {
            "feature": np.asarray(feature, dtype=np.int64),
            "threshold": np.asarray(threshold, dtype=np.float64),
            "left": np.asarray(left, dtype=np.int64),
            "right": np.asarray(right, dtype=np.int64),
            "value_start": np.asarray(start, dtype=np.int64),
            "value_count": np.asarray(count, dtype=np.int64),
            "values": np.concatenate(values) if values else np.zeros(0),
            "roots": np.asarray(roots, dtype=np.int64),
            "feature_importances": np.asarray(self.feature_importances_, dtype=np.float64),
        }
        meta = {name: getattr(self, name) for name in self._PARAMS}
        meta.update(n_features_in=int(self.n_features_in_), n_samples_bootstrap=int(self.n_samples_bootstrap_))
        return meta, arrays

    @classmethod
    def from_arrays(cls, meta: dict, arrays) -> CRPSForest:
        forest = cls(**{name: meta[name] for name in cls._PARAMS})
        forest.n_features_in_ = int(meta["n_features_in"])
        forest.n_samples_bootstrap_ = int(meta["n_samples_bootstrap"])
        feature, threshold = arrays["feature"], arrays["threshold"]
        left, right = arrays["left"], arrays["right"]
        start, count, values = arrays["value_start"], arrays["value_count"], arrays["values"]

        def node(i):
            if feature[i] < 0:
                return _Node(values=values[start[i]:start[i] + count[i]].copy())
            return _Node(feature=int(feature[i]), threshold=float(threshold[i]), left=node(int(left[i])),
                         right=node(int(right[i])))

        trees = []
        for root in arrays["roots"]:
            tree = _CRPSTree(min_samples_leaf=forest.min_samples_leaf, max_depth=forest.max_depth,
                             max_features=forest.max_features)
            tree.n_features_in_ = forest.n_features_in_
            tree.root_ = node(int(root))
            trees.append(tree)
        forest.trees_ = trees
        forest.feature_importances_ = np.asarray(arrays["feature_importances"], dtype=np.float64)
        return forest
