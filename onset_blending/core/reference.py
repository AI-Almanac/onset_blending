"""The reference data a blend reads besides rainfall, resolved once.

Thresholds, reference onset dates, cell transforms and regrid weights are
read from the config's files when a blend is trained, held here as Python
objects, and stored with the trained blend, so a forecast uses exactly what
training used without reading (or writing) any file.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import domain, ids
from ..config import BlendConfig

log = logging.getLogger(__name__)

OBSERVATIONS = "observations"


@dataclass
class Reference:
    thresholds: float | pd.DataFrame | None
    reference_onset: dict | pd.DataFrame | None
    cell_transforms: dict[str, pd.DataFrame] = field(default_factory=dict)  # by source: observations or a model
    regrid_weights: dict[str, pd.DataFrame] = field(default_factory=dict)
    _regridders: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def resolve(cls, config: BlendConfig, *, workers: int = 1, cache_dir=None) -> "Reference":
        """Read (or, for onset.threshold.climatology, compute from the
        observations, kept in cache_dir when given) everything the stages
        look up."""
        cell_transforms, regrid_weights = {}, {}
        sources = [(OBSERVATIONS, config.observations.cell_transform, None)] + [
            (m.name, m.cell_transform, m) for m in config.models
        ]
        for name, transform, model in sources:
            if transform is not None:
                cell_transforms[name] = read_cell_transform(config.path(transform))
            weights = _regrid_weights_path(config, model)
            if weights is not None:
                regrid_weights[name] = pd.read_csv(weights, dtype={"target_id": str, "adm3_name": str})
        reference = cls(
            thresholds=read_thresholds(config),
            reference_onset=read_reference_onset(config),
            cell_transforms=cell_transforms,
            regrid_weights=regrid_weights,
        )
        if config.onset.threshold.climatology is not None:
            from .onsets import climatological_thresholds

            computed = climatological_thresholds(config, reference, workers=workers, cache_dir=cache_dir)
            reference.thresholds = _scaled(computed, config.onset.threshold.scale)
            log.info("Computed onset thresholds for %d units from the observations", len(reference.thresholds))
        return reference

    def unit_thresholds(self, unit_ids: np.ndarray, explicit: ids.GridIdConvention | None) -> np.ndarray:
        """Each unit's onset threshold (NaN without one); a table keyed by
        lat/lon is read under the units' id convention, and must cover them."""
        thresholds = self.thresholds
        if thresholds is None:
            return np.full(len(unit_ids), np.nan)
        if not isinstance(thresholds, pd.DataFrame):
            return np.full(len(unit_ids), float(thresholds))
        convention = ids.resolve(explicit, authoritative_ids=pd.Series(unit_ids, dtype=str),
                                 context="rainfall/threshold ID handoff")
        table = thresholds.copy()
        table.columns = table.columns.str.lower()
        if "id" not in table.columns:
            if "adm3_name" in table.columns:
                table["id"] = table["adm3_name"]
            elif {"lat", "lon"} <= set(table.columns):
                coords = ids.with_id_column(table[["lat", "lon"]], explicit=explicit, convention=convention,
                                            context="latitude/longitude IDs")
                table["id"] = coords["id"].to_numpy()
            else:
                raise ValueError("Thresholds table must have 'id', 'adm3_name', or ('lat', 'lon').")
        table["id"] = ids.normalize(table["id"], context="threshold IDs")
        table["onset_thresh"] = table["onset_thresh"].astype(float)
        table = table[["id", "onset_thresh"]].drop_duplicates()
        conflicting = table["id"].duplicated(keep=False)
        if conflicting.any():
            raise ValueError(f"Threshold IDs have conflicting values: {', '.join(table.loc[conflicting, 'id'].unique()[:10])}")
        by_id = table.set_index("id")["onset_thresh"]
        wanted = pd.Index(pd.unique(np.asarray(unit_ids, dtype=str)))
        missing = wanted[~wanted.isin(by_id.index)]
        if len(missing):
            raise ValueError(
                f"Threshold table is missing {len(missing)} of {len(wanted)} rainfall IDs after domain "
                f"filtering. Examples: {', '.join(map(str, missing[:10]))}"
            )
        return by_id.reindex(np.asarray(unit_ids, dtype=str)).to_numpy(dtype=float)

    def reference_dates(self, unit_ids: np.ndarray, years: np.ndarray) -> np.ndarray:
        """The reference onset date (datetime64[ns], NaT without one) of each
        (unit, year) pair."""
        years = np.asarray(years, dtype=int)
        reference = self.reference_onset
        if reference is None:
            return np.full(len(years), np.datetime64("NaT"), dtype="datetime64[ns]")
        if isinstance(reference, dict):
            month_day = reference["month_day"]
            distinct, codes = np.unique(years, return_inverse=True)
            dates = pd.to_datetime(pd.Series([f"{y}-{month_day}" for y in distinct]), errors="coerce").to_numpy()
            return dates[codes].astype("datetime64[ns]")
        keys = [k for k in ("id", "year") if k in reference.columns]
        if not keys:
            return np.full(len(years), np.datetime64("NaT"), dtype="datetime64[ns]")
        table = reference.drop_duplicates(keys + ["ref_onset_date"])
        if table.duplicated(keys).any():
            raise ValueError("The reference onset table gives a unit and year more than one date")
        pairs = pd.DataFrame({"id": pd.Series(np.asarray(unit_ids, dtype=str)), "year": years})
        found = pairs[keys].merge(table, on=keys, how="left")["ref_onset_date"]
        return pd.to_datetime(found).to_numpy().astype("datetime64[ns]")

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


def read_thresholds(config: BlendConfig) -> float | pd.DataFrame | None:
    """The onset threshold: a constant, or a table (lat, lon, onset_thresh or
    id, onset_thresh) from a CSV, NetCDF or MATLAB file, optionally scaled;
    None for one computed from the observations (Reference.resolve does)."""
    threshold = config.onset.threshold
    if threshold.mm is not None:
        return float(threshold.mm)
    if threshold.climatology is not None:
        return None
    path = str(config.path(threshold.file))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Threshold file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nc", ".nc4", ".netcdf"):
        import netCDF4

        with netCDF4.Dataset(path) as ds:
            table = pd.DataFrame({
                "lat": np.array(ds[threshold.lat_var][:]).flatten(),
                "lon": np.array(ds[threshold.lon_var][:]).flatten(),
                "onset_thresh": np.array(ds[threshold.value_var][:]).flatten(),
            }).drop_duplicates()
    elif ext == ".mat":
        import scipy.io

        content = scipy.io.loadmat(path)
        lats = np.array(content[threshold.mat_lat_name]).flatten()
        lons = np.array(content[threshold.mat_lon_name]).flatten()
        table = pd.DataFrame([(a, b) for a in lats for b in lons], columns=["lat", "lon"])
        table["onset_thresh"] = np.array(content[threshold.mat_value_name]).flatten()
        table = table.drop_duplicates()
    else:
        table = _threshold_csv(path, threshold)
    return _scaled(table, threshold.scale)


def _scaled(table: pd.DataFrame, scale) -> pd.DataFrame:
    """clip(threshold * factor + offset, min, max), or the table unchanged."""
    if scale is None:
        return table
    values = table["onset_thresh"].astype(float).values * float(scale.factor) + float(scale.offset)
    if scale.min is not None:
        values = np.maximum(values, float(scale.min))
    if scale.max is not None:
        values = np.minimum(values, float(scale.max))
    return table.assign(onset_thresh=values)


def _threshold_csv(path: str, threshold) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str)
    value = threshold.value_col
    if threshold.id_col:
        if threshold.id_col not in table.columns:
            raise ValueError(f"Configured thresholds.id_col '{threshold.id_col}' not found. Found: {table.columns.tolist()}")
        return table.rename(columns={threshold.id_col: "id", value: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    if "adm3_name" in table.columns:
        return table.rename(columns={"adm3_name": "id", value: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    if "id" in table.columns:
        return table.rename(columns={value: "onset_thresh"})[["id", "onset_thresh"]].drop_duplicates()
    table = table.rename(columns={threshold.lat_col: "lat", threshold.lon_col: "lon", value: "onset_thresh"})
    return table[["lat", "lon", "onset_thresh"]].drop_duplicates()


def read_reference_onset(config: BlendConfig) -> dict | pd.DataFrame | None:
    """The reference onset: None, a month-day every year and unit
    ({"mode": "constant_month_day", ...}), or a table of ref_onset_date
    keyed by id and/or year."""
    reference = config.onset.reference_onset
    if reference is None:
        return None
    if reference.month_day is not None:
        return {"mode": "constant_month_day", "month_day": str(reference.month_day)}
    path = str(config.path(reference.file))
    if not os.path.exists(path):
        raise FileNotFoundError(f"ref_onset file not found: {path}")
    unit_col, year_col, day_col = reference.unit_col, reference.year_col, reference.day_col
    table = pd.read_csv(path, dtype={unit_col: str} if unit_col else None)
    if reference.date_col and reference.date_col in table.columns:
        table["ref_onset_date"] = pd.to_datetime(table[reference.date_col], errors="coerce").dt.date
    elif year_col and day_col and reference.base_date and year_col in table.columns and day_col in table.columns:
        table["ref_onset_date"] = table.apply(
            lambda row: (pd.to_datetime(f"{int(row[year_col])}-{reference.base_date}") + pd.Timedelta(days=int(row[day_col]))).date(),
            axis=1,
        )
    else:
        raise ValueError("ref_onset file needs either a parseable 'date_col', or 'year_col'+'day_col'+'base_date'.")
    keys = []
    if unit_col:
        if unit_col not in table.columns:
            raise ValueError(f"ref_onset unit_col '{unit_col}' not in file columns {table.columns.tolist()}")
        table = table.rename(columns={unit_col: "id"})
        table["id"] = ids.normalize(table["id"], context="reference-onset IDs")
        keys.append("id")
    if year_col and year_col in table.columns:
        table = table.rename(columns={year_col: "year"})
        table["year"] = table["year"].astype(int)
        keys.append("year")
    if not keys:
        raise ValueError("ref_onset file needs a 'year_col' and/or 'unit_col' to key on.")
    return table[keys + ["ref_onset_date"]].drop_duplicates()


def read_cell_transform(path) -> pd.DataFrame:
    """Source-to-unit weights (source_id, target_id, weight): normalised ids,
    finite non-negative weights, distinct pairs, every target weighted; rows
    of zero weight dropped."""
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"cell_transform_file not found: {path}")
    weights = pd.read_csv(path, dtype={"source_id": str, "target_id": str})
    if any(c not in weights.columns for c in ("target_id", "source_id", "weight")):
        raise ValueError("Transform file must have: target_id, source_id, weight")
    if weights.empty:
        raise ValueError("Cell-transform weights file must not be empty.")
    weights["source_id"] = ids.normalize(weights["source_id"], context="cell-transform source IDs")
    weights["target_id"] = ids.normalize(weights["target_id"], context="cell-transform target IDs")
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    if not np.isfinite(weights["weight"]).all() or (weights["weight"] < 0).any():
        raise ValueError("Cell-transform weights must be finite and non-negative.")
    if weights.duplicated(["source_id", "target_id"]).any():
        raise ValueError("Cell-transform source_id/target_id pairs must be unique.")
    totals = weights.groupby("target_id")["weight"].sum()
    if (totals <= 0).any():
        raise ValueError(f"Cell-transform targets have zero total weight: {', '.join(totals[totals <= 0].index.astype(str)[:10])}")
    return weights.loc[weights["weight"] > 0].reset_index(drop=True)


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
