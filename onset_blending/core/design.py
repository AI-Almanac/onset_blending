"""Design matrices from formulas, with array math instead of tables.

A formula whose terms are numeric columns and products of them (`a:b`, or
`a*b` expanded) makes every design column a product of feature columns.
`plan` takes the formula's columns, in Patsy's order, from Patsy itself;
`Plan.build` then fills the matrix a column at a time straight from the
feature arrays, multiplying the factors in Patsy's order, so the values
are Patsy's to the bit without a table or Patsy's intermediate copies.
Other formulas (functions, categorical or stateful terms) have no plan
and go through Patsy.

The formulas themselves come from the config here too: `formulas` expands
each enabled text (`_qx` to one term per week bin, `a*b` to every
interaction), drops the `_qx` terms the rain horizon cannot build, and adds
the window variants; `sample_support` gives the samples every formula's
predictors are finite for.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy


def right_hand_side(formula: str) -> str:
    return formula.split("~", 1)[1].strip() if "~" in formula else formula.strip()


@dataclass(frozen=True)
class Plan:
    """A formula's design columns, each the product of named feature columns."""

    formula: str
    columns: tuple[str, ...]
    factors: tuple[tuple[str, ...], ...]

    @property
    def features(self) -> list[str]:
        """The feature columns the design reads, in order of first use."""
        return list(dict.fromkeys(name for names in self.factors for name in names))

    def column(self, j: int, values: Callable[[str], np.ndarray], rows: np.ndarray | None = None) -> np.ndarray:
        """Design column j, for `rows` (positions) or every row: its factors'
        product, multiplied in Patsy's order."""
        out = None
        for name in self.factors[j]:
            factor = np.asarray(values(name), dtype=float)
            if rows is not None:
                factor = factor[rows]
            if out is None:
                out = factor.copy() if rows is None else factor
            else:
                out *= factor
        return out

    def build(self, values: Callable[[str], np.ndarray], rows: np.ndarray | None = None, *,
              out: np.ndarray | None = None, order: str = "F") -> np.ndarray:
        """The design matrix (rows, columns), into `out` when given.

        values(name) gives a feature's values per row of the source."""
        if out is None:
            n = len(rows) if rows is not None else len(np.asarray(values(self.factors[0][0])))
            out = np.empty((n, len(self.columns)), order=order)
        for j in range(len(self.columns)):
            out[:, j] = self.column(j, values, rows)
        return out


def plan(formula: str, dtypes: Mapping[str, object]) -> Plan | None:
    """The design of a formula whose every term is a product of numeric
    columns of `dtypes` (column name -> dtype); None for any other formula."""
    rhs = right_hand_side(formula)
    try:
        terms = patsy.ModelDesc.from_formula(rhs).rhs_termlist
    except patsy.PatsyError:
        return None
    by_name = {}
    for term in terms:
        names = []
        for factor in term.factors:
            name = getattr(factor, "code", None)
            if name not in dtypes or not _numeric(dtypes[name]):
                return None
            names.append(name)
        if names:
            by_name[term.name()] = tuple(names)
    if not by_name:
        return None
    # Patsy orders the columns (it groups and sorts terms): ask it, on one row.
    used = sorted({name for names in by_name.values() for name in names})
    sample = pd.DataFrame({name: np.zeros(1, dtype=dtypes[name]) for name in used})
    columns = [c for c in patsy.dmatrix(rhs, sample, return_type="dataframe").columns if c != "Intercept"]
    if len(columns) != len(by_name) or set(columns) != set(by_name):
        return None
    return Plan(formula, tuple(columns), tuple(by_name[c] for c in columns))


def _numeric(dtype) -> bool:
    """Patsy treats these as numeric (not bool, object or extension types)."""
    return isinstance(dtype, np.dtype) and dtype.kind in "fiu"


# ---------------------------------------------------------------------------
# Formulas: the config's texts, expanded and resolved against the features
# ---------------------------------------------------------------------------


def formula_columns(formula: str) -> list[str]:
    """The right-hand side's feature names (operators and constants stripped)."""
    rhs = formula.split("~", 1)[1] if "~" in formula else formula
    names = [t.strip() for t in re.split(r"[+\-\*:\(\)]", rhs)]
    return [n for n in names if n and not n.isdigit() and n != "1"]


def _expanded_terms(formula: str, n_bins: int) -> tuple[str, list[tuple[str, set[str]]]]:
    """(lhs, [(term, the _qx-generated columns it uses)]): `_qx` terms expand
    to one per week bin, `a*b` to every Wilkinson interaction."""
    if "~" in formula:
        lhs, rhs = (part.strip() for part in formula.split("~", 1))
    else:
        lhs, rhs = "", formula.strip()
    by_week = []
    for raw in (t.strip() for t in rhs.split("+")):
        qx = [c for c in formula_columns(raw) if "_qx" in c]
        if qx:
            for i in range(1, int(n_bins) + 1):
                by_week.append((raw.replace("_qx", f"_week{i}"), {c.replace("_qx", f"_week{i}") for c in qx}))
        else:
            by_week.append((raw, set()))
    terms = []
    for term, generated in by_week:
        if "*" in term:
            factors = [v.strip() for v in term.split("*")]
            for size in range(1, len(factors) + 1):
                for combo in itertools.combinations(factors, size):
                    terms.append((":".join(combo), generated.intersection(combo)))
        else:
            terms.append((term, generated))
    return lhs, terms


def _joined(lhs: str, terms: list[str]) -> str:
    rhs = " + ".join(terms)
    return f"{lhs} ~ {rhs}" if lhs else rhs


def expand(formula: str, n_bins: int) -> str:
    """The formula with `_qx` and `*` expanded."""
    lhs, terms = _expanded_terms(formula, n_bins)
    return _joined(lhs, [term for term, _ in terms])


def resolve(formula: str, unavailable, n_bins: int) -> tuple[str, list[str]]:
    """(the expanded formula without the `_qx` terms whose columns cannot be
    built under the rain horizon, the dropped terms). Explicit `_weekN` terms
    are never dropped."""
    lhs, terms = _expanded_terms(formula, n_bins)
    unavailable = set(unavailable)
    kept, dropped = [], []
    for term, generated in terms:
        (dropped if generated & unavailable else kept).append(term)
    if not any(formula_columns(term) for term in kept):
        raise ValueError("Formula has no constructible predictor terms after resolving connector rainfall horizons.")
    return _joined(lhs, kept), dropped


def formulas(evaluation, unavailable, n_bins: int, cutoff_mode: str) -> tuple[dict[str, str], dict[str, dict]]:
    """The enabled formulas, resolved, with their window variants, and how
    each was resolved (config.Evaluation: formulas, window_variants).

    A window variant is its base formula with the `<replace>_qx` feature
    swapped for its windowed version `<replace>_<start>_<end>_qx`, resolved
    as a formula of its own. (The legacy evaluation swapped it in the
    expanded text, where `_qx` no longer occurs, so every variant repeated
    its base formula.)"""
    if not evaluation.formulas:
        raise ValueError("Spec must define models.formulas with named entries containing 'text'.")
    out, resolution = {}, {}

    def add(name, text):
        resolved, dropped = resolve(text, unavailable, n_bins)
        out[name] = resolved
        resolution[name] = {
            "original_formula": text,
            "expanded_formula": expand(text, n_bins),
            "resolved_formula": resolved,
            "dropped_terms": dropped,
        }

    texts = {}
    for name, entry in evaluation.formulas.items():
        if not entry.enabled:
            continue
        if not entry.text:
            raise ValueError(f"Formula '{name}' is enabled but has no non-empty 'text' in spec.")
        texts[name] = entry.text
        add(name, entry.text)
    for variant in evaluation.window_variants:
        if variant.only_if_cutoff_mode and cutoff_mode != variant.only_if_cutoff_mode:
            continue
        if variant.base not in texts:
            raise ValueError("window_variants.base_name must match a name in models.formulas.")
        feature = re.compile(rf"\b{re.escape(variant.replace)}_qx\b")
        for start, end in variant.windows:
            add(f"{variant.base}_{start}_{end}", feature.sub(f"{variant.replace}_{start}_{end}_qx", texts[variant.base]))
    return out, resolution


def _as_float(values) -> np.ndarray:
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)


def sample_support(weekly, formulas: dict[str, str]) -> tuple[np.ndarray, dict]:
    """The samples every enabled formula's predictors are finite for
    (positions), and diagnostics; refuses a predictor that is missing or
    never finite (naming a rain-horizon cause when the features recorded one)."""
    feature_map = {name: list(dict.fromkeys(formula_columns(text))) for name, text in formulas.items()}
    unavailable = weekly.attrs.get("unavailable_rain_predictors", {})
    problems = []
    for name, columns in feature_map.items():
        missing = [c for c in columns if c not in weekly.columns]
        if missing:
            problems.append(f"Formula '{name}' is missing predictor columns: {', '.join(missing)}")
        for column in columns:
            if column not in weekly.columns or np.isfinite(_as_float(weekly[column])).any():
                continue
            details = unavailable.get(column)
            if details:
                problems.append(
                    f"Formula '{name}': {column} cannot be constructed under rain_day_max={details['rain_day_max']} "
                    f"(window={details['window']}, week={details['week']}; no complete start days)."
                )
            else:
                problems.append(f"Formula '{name}': {column} has no finite values and cannot be used for fitting.")
    if problems:
        raise ValueError("Formula feature validation failed before model fitting:\n- " + "\n- ".join(problems))
    required = list(dict.fromkeys(c for columns in feature_map.values() for c in columns))
    keep = np.ones(len(weekly), dtype=bool)
    excluded = {}
    for column in required:
        finite = np.isfinite(_as_float(weekly[column]))
        excluded[column] = int((~finite).sum())
        keep &= finite
    diagnostics = {
        "policy": "common_complete",
        "required_columns": required,
        "rows_before": int(len(weekly)),
        "rows_after": int(keep.sum()),
        "rows_excluded": int((~keep).sum()),
        "nonfinite_by_column": {c: n for c, n in excluded.items() if n},
    }
    return np.flatnonzero(keep), diagnostics

