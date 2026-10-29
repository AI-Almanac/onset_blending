"""Design matrices from formulas, with array math instead of tables.

A formula whose terms are numeric columns and products of them (`a:b`, or
`a*b` expanded) makes every design column a product of feature columns.
`plan` takes the formula's columns, in Patsy's order, from Patsy itself;
`Plan.build` then fills the matrix a column at a time straight from the
feature arrays, multiplying the factors in Patsy's order, so the values
are Patsy's to the bit without a table or Patsy's intermediate copies.
Other formulas (functions, categorical or stateful terms) have no plan
and go through Patsy.
"""

from __future__ import annotations

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
