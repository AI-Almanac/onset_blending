"""The reference data a blend reads besides rainfall, resolved once.

Thresholds, reference onset dates, cell transforms and regrid weights are
read from the config's files when a blend is trained, held here as Python
objects, and stored with the trained blend, so a forecast uses exactly what
training used without reading (or writing) any file.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import _specs, domain
from ._legacy import nc_utils, onset_utils
from .config import BlendConfig

OBSERVATIONS = "observations"


@dataclass
class Reference:
    thresholds: float | pd.DataFrame | None
    reference_onset: dict | pd.DataFrame | None
    cell_transforms: dict[str, pd.DataFrame] = field(default_factory=dict)  # by source: observations or a model
    regrid_weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    _regridders: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def resolve(cls, config: BlendConfig) -> "Reference":
        spec = _specs.observation_spec(config)
        cell_transforms, regrid_weights = {}, {}
        sources = [(OBSERVATIONS, spec, None)] + [
            (m.name, _specs.forecast_spec(config, m), m) for m in config.models
        ]
        for name, source_spec, model in sources:
            transform = nc_utils.read_cell_transform(source_spec)
            if transform is not None:
                cell_transforms[name] = transform
            weights = _regrid_weights_path(config, model)
            if weights is not None:
                regrid_weights[name] = pd.read_csv(weights, dtype={"target_id": str, "adm3_name": str})
        return cls(
            thresholds=onset_utils.read_thresholds(spec),
            reference_onset=onset_utils.read_ref_onset_dates(spec),
            cell_transforms=cell_transforms,
            regrid_weights=regrid_weights,
        )

    def regridder(self, source: str) -> domain.Regridder | None:
        """The regridder onto the blend's units for a source, compiled once."""
        if source not in self.regrid_weights:
            return None
        if source not in self._regridders:
            self._regridders[source] = domain.Regridder(self.regrid_weights[source])
        return self._regridders[source]

    # -- without pickles ------------------------------------------------------

    def to_arrays(self) -> tuple[dict, dict[str, np.ndarray]]:
        """JSON metadata and plain arrays (strings as unicode, dates as ISO text)."""
        meta, arrays = {}, {}
        if isinstance(self.thresholds, pd.DataFrame):
            meta["thresholds"] = _put(arrays, "thresholds", self.thresholds)
        else:
            meta["thresholds"] = self.thresholds
        if isinstance(self.reference_onset, pd.DataFrame):
            meta["reference_onset"] = {"table": _put(arrays, "reference_onset", self.reference_onset)}
        else:
            meta["reference_onset"] = self.reference_onset
        meta["cell_transforms"] = {k: _put(arrays, f"cell_{k}", v) for k, v in self.cell_transforms.items()}
        meta["regrid_weights"] = {k: _put(arrays, f"regrid_{k}", v) for k, v in self.regrid_weights.items()}
        return meta, arrays

    @classmethod
    def from_arrays(cls, meta: dict, arrays) -> "Reference":
        thresholds = meta["thresholds"]
        if isinstance(thresholds, dict):
            thresholds = _get(arrays, thresholds)
        reference_onset = meta["reference_onset"]
        if isinstance(reference_onset, dict) and "table" in reference_onset:
            reference_onset = _get(arrays, reference_onset["table"])
        return cls(
            thresholds=thresholds,
            reference_onset=reference_onset,
            cell_transforms={k: _get(arrays, v) for k, v in meta["cell_transforms"].items()},
            regrid_weights={k: _get(arrays, v) for k, v in meta["regrid_weights"].items()},
        )


def _regrid_weights_path(config: BlendConfig, model):
    regrid = config.domain.regrid
    if regrid is None:
        return None
    supplied = (model.regrid_weights if model is not None else None) or regrid.weights
    if supplied is None:
        raise NotImplementedError(
            "Building regrid weights from domain.regrid.shapefile is not yet supported; "
            "give domain.regrid.weights"
        )
    return config.path(supplied)


def _put(arrays: dict, prefix: str, table: pd.DataFrame) -> dict:
    """A table's columns into arrays; returns its schema."""
    columns = []
    for index, name in enumerate(table.columns):
        series, key = table[name], f"{prefix}.{index}"
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            kind, arrays[key] = "number", series.to_numpy()
        else:
            values = series.astype(object)
            is_date = values.map(lambda v: isinstance(v, dt.date) and not pd.isna(v))
            kind = "date" if is_date.any() else "text"
            arrays[key] = np.array(
                [v.isoformat() if kind == "date" and d else ("" if pd.isna(v) else str(v)) for v, d in zip(values, is_date)],
                dtype=str,
            )
            arrays[f"{key}.missing"] = values.isna().to_numpy()
        columns.append({"name": str(name), "kind": kind, "dtype": str(series.dtype)})
    return {"prefix": prefix, "rows": len(table), "columns": columns}


def _get(arrays, schema: dict) -> pd.DataFrame:
    data = {}
    for index, column in enumerate(schema["columns"]):
        key = f"{schema['prefix']}.{index}"
        values = arrays[key]
        if column["kind"] == "number":
            data[column["name"]] = pd.Series(values, dtype=column["dtype"])
            continue
        missing = arrays[f"{key}.missing"]
        if column["kind"] == "date":
            items = [None if m else dt.date.fromisoformat(v) for v, m in zip(values.tolist(), missing.tolist())]
            data[column["name"]] = pd.Series(items, dtype=object)
        else:
            items = [None if m else v for v, m in zip(values.tolist(), missing.tolist())]
            data[column["name"]] = pd.Series(items, dtype=column["dtype"])
    return pd.DataFrame(data, index=pd.RangeIndex(schema["rows"]))
