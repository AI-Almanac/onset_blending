"""Where a blend applies: unit ids, the area of interest, dissemination units,
and regridding inputs onto the units."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
from scipy import sparse

from . import ids as unit_ids
from ..config import BlendConfig
from .ids import GridIdConvention, format_coord_component

log = logging.getLogger(__name__)


def id_convention(config: BlendConfig) -> GridIdConvention | None:
    """The configured "<lat>_<lon>" convention; None means inferred per table."""
    return unit_ids.configured(config)


def with_ids(df: pd.DataFrame, convention: GridIdConvention | None) -> pd.DataFrame:
    """df with an `id` column: kept if present, else "<lat>_<lon>" from each
    row's coordinates. Each distinct coordinate is formatted once; a grid has
    thousands of cells where its rows run to tens of millions."""
    if "id" in df.columns:
        return df
    if convention is None or not {"lat", "lon"} <= set(df.columns):
        return unit_ids.with_id_column(df, context="input spatial IDs")
    lat_codes, lats = pd.factorize(df["lat"].to_numpy(dtype=np.float64))
    lon_codes, lons = pd.factorize(df["lon"].to_numpy(dtype=np.float64))
    lat_text = np.array([format_coord_component(v, convention) for v in lats], dtype=object)
    lon_text = np.array([format_coord_component(v, convention) for v in lons], dtype=object)
    pair_codes, pairs = pd.factorize(lat_codes * len(lons) + lon_codes)
    labels = np.array(
        [f"{lat_text[p // len(lons)]}_{lon_text[p % len(lons)]}" for p in pairs], dtype=object
    )
    return df.assign(id=pd.Series(labels[pair_codes], index=df.index).astype(str))


class AreaFilter:
    """Keeps rows inside the area of interest (a box, or GeoJSON outlines) and,
    when the dissemination units restrict the inputs, only those units."""

    def __init__(self, config: BlendConfig):
        domain = config.domain
        self.convention = unit_ids.configured(config)
        self.centroids = None
        if domain.centroids is not None:
            self.centroids = (str(config.path(domain.centroids.file)), domain.centroids.id_col)
        area = domain.area_of_interest
        self.bbox = None
        self.outlines = None
        if area is not None and area.bbox is not None:
            self.bbox = {k: v for k, v in vars(area.bbox).items() if v is not None}
        elif area is not None:
            with open(config.path(area.geometry)) as handle:
                self.outlines = json.load(handle)
        self.dissemination = None
        if domain.dissemination is not None and domain.dissemination_scope == "inputs":
            self.dissemination = (str(config.path(domain.dissemination.file)), domain.dissemination.id_col)

    @classmethod
    def everything(cls) -> "AreaFilter":
        """No area of interest and no dissemination filter (a trained blend's
        units are given instead)."""
        area = cls.__new__(cls)
        area.convention = area.centroids = area.bbox = area.outlines = area.dissemination = None
        return area

    @property
    def active(self) -> bool:
        return bool(self.bbox or self.outlines or self.dissemination)

    def area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows inside the area of interest."""
        if self.bbox:
            df = unit_ids.with_id_column(df, explicit=self.convention, context="rainfall spatial IDs")
            lat, lon = self._lat_lon(df)
            if lat is None:
                raise ValueError(
                    "filter.bbox requested but unit lat/lon could not be determined: "
                    "df has no 'lat'/'lon' columns, ids are not '<lat>_<lon>', and no "
                    "filter.centroids_file was provided."
                )
            mask = np.ones(len(df), dtype=bool)
            for key, test in (("lat_min", lambda v: lat >= v), ("lat_max", lambda v: lat <= v),
                              ("lon_min", lambda v: lon >= v), ("lon_max", lambda v: lon <= v)):
                if self.bbox.get(key) is not None:
                    mask &= test(float(self.bbox[key]))
            kept = df[mask]
            log.info("bbox filter %s: %d -> %d rows", self.bbox, len(df), len(kept))
        elif self.outlines:
            kept = self._inside_outlines(df)
        else:
            return df
        if kept.empty:
            raise ValueError("No cells fall inside the area of interest")
        return kept

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.area(df)
        if self.dissemination:
            df = self._in_dissemination(df)
        return df

    def _in_dissemination(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows of the dissemination units, their ids read under the rows' convention."""
        df = unit_ids.with_id_column(df, explicit=self.convention, context="rainfall spatial IDs")
        path, id_col = self.dissemination
        if not os.path.exists(path):
            raise FileNotFoundError(f"dissemination_cells_file not found: {path}")
        cells = pd.read_csv(path, dtype=str)
        if id_col:
            if id_col not in cells.columns:
                raise ValueError(f"Configured dissemination id_col '{id_col}' not found. Found: {cells.columns.tolist()}")
            cells = cells.rename(columns={id_col: "id"})
        elif not any(c in cells.columns for c in ("id", "adm3_name")) and not {"lat", "lon"} <= set(cells.columns):
            log.warning("dissemination_cells_file has no explicit spatial key; using its first column %r",
                        cells.columns[0])
            cells = cells.rename(columns={cells.columns[0]: "id"})
        convention = unit_ids.resolve(self.convention, authoritative_ids=df["id"],
                                      context="rainfall/dissemination ID handoff")
        cells = unit_ids.with_id_column(cells, explicit=self.convention, convention=convention,
                                        context="dissemination cell IDs")
        unit_ids.check_coordinates(cells, context="dissemination cell IDs")
        kept = df[df["id"].isin(set(cells["id"]))]
        log.info("dissemination_cells filter: %d -> %d rows", len(df), len(kept))
        return kept

    def _lat_lon(self, df: pd.DataFrame):
        """Each row's (lat, lon): its columns, its "<lat>_<lon>" id, or its
        unit's centroid; (None, None) without any."""
        if "lat" in df.columns and "lon" in df.columns:
            return df["lat"].astype(float).values, df["lon"].astype(float).values
        parsed = df["id"].astype(str).str.extract(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
        if len(df) and parsed.notna().all(axis=None):
            return parsed[0].astype(float).values, parsed[1].astype(float).values
        if self.centroids is None:
            return None, None
        path, id_col = self.centroids
        if not os.path.exists(path):
            raise FileNotFoundError(f"filter.centroids_file not found: {path}")
        centroids = pd.read_csv(path, dtype=str)
        if id_col and id_col not in centroids.columns:
            raise ValueError(f"filter.centroids_id_col '{id_col}' not found. Found: {centroids.columns.tolist()}")
        key = id_col or next((c for c in ("target_id", "id", "adm3_name") if c in centroids.columns),
                             centroids.columns[0])
        lat = next((c for c in ("lat", "center_lat", "latitude") if c in centroids.columns), None)
        lon = next((c for c in ("lon", "center_lon", "longitude") if c in centroids.columns), None)
        if not lat or not lon:
            raise ValueError("filter.centroids_file must have lat/lon (or center_lat/center_lon).")
        centroids = centroids.rename(columns={key: "id", lat: "_lat", lon: "_lon"})
        centroids["id"] = unit_ids.normalize(centroids["id"], context="centroid IDs")
        merged = df[["id"]].merge(centroids[["id", "_lat", "_lon"]].drop_duplicates("id"), on="id", how="left")
        return merged["_lat"].astype(float).values, merged["_lon"].astype(float).values

    def _inside_outlines(self, df: pd.DataFrame) -> pd.DataFrame:
        import shapely
        from shapely.geometry import shape

        lat, lon = self._lat_lon(df)
        if lat is None:
            raise ValueError("The area of interest needs a lat/lon per row, or domain.centroids")
        outline = shapely.union_all([shape(f["geometry"]) for f in self.outlines["features"]])
        return df[shapely.contains_xy(outline, lon, lat)]


class Regridder:
    """Area-weighted means of gridded values over the blend's units: the
    sparse transform the legacy regrid applied file by file (a weighted mean
    over each unit's observed source cells), applied to arrays as read."""

    def __init__(self, weights: pd.DataFrame):
        mapping = regrid_weights(weights, context="regrid weights")
        mapping = mapping.rename(columns={"latitude": "lat", "longitude": "lon"})
        for column in ("lat", "lon", "weight"):
            mapping[column] = pd.to_numeric(mapping[column], errors="raise")
        mapping["source_id"] = list(zip(mapping["lat"], mapping["lon"]))
        mapping["target_id"] = mapping["target_id"].astype(str)
        self.mapping = mapping
        self.targets = np.asarray(sorted(mapping["target_id"].unique()), dtype=object)
        self._transforms: dict[tuple, object] = {}

    @classmethod
    def from_csv(cls, path) -> "Regridder":
        return cls(pd.read_csv(path, dtype={"target_id": str, "adm3_name": str}))

    def transform(self, lats: np.ndarray, lons: np.ndarray):
        """The compiled transform for a grid (latitude-major source order)."""
        sources = tuple((float(a), float(b)) for a in lats for b in lons)
        if sources not in self._transforms:
            self._transforms[sources] = SparseTransform.compile(
                self.mapping, source_ids=sources, target_ids=tuple(self.targets)
            )
        return self._transforms[sources]

    @staticmethod
    def sources_of(transform, targets: np.ndarray) -> np.ndarray:
        """Source positions with weight for the given target positions."""
        return np.unique(transform.weighted_matrix[targets].indices)

    @staticmethod
    def apply(transform, values: np.ndarray) -> np.ndarray:
        """(source, column) values -> (target, column) weighted means; NaN
        sources are unobserved."""
        return transform.observed_mean(values)


def regrid_weights(weights: pd.DataFrame, context: str = "regrid weights") -> pd.DataFrame:
    """A weight table (target_id or legacy adm3_name, lat/lon or
    latitude/longitude, weight) with normalised target ids in both columns,
    refusing duplicate (cell, target) rows."""
    out = weights.copy()
    if "target_id" not in out.columns and "adm3_name" not in out.columns:
        raise ValueError(f"{context} must contain 'target_id' or 'adm3_name'.")
    if "target_id" in out.columns:
        out["target_id"] = unit_ids.normalize(out["target_id"], context=f"{context} target IDs")
    if "adm3_name" in out.columns:
        out["adm3_name"] = unit_ids.normalize(out["adm3_name"], context=f"{context} legacy target IDs")
    if {"target_id", "adm3_name"} <= set(out.columns):
        if (out["target_id"] != out["adm3_name"]).any():
            raise ValueError(f"{context}: 'target_id' and 'adm3_name' disagree.")
    elif "target_id" in out.columns:
        out["adm3_name"] = out["target_id"]
    else:
        out["target_id"] = out["adm3_name"]
    lat = "latitude" if "latitude" in out.columns else "lat"
    lon = "longitude" if "longitude" in out.columns else "lon"
    if any(column not in out.columns for column in (lat, lon, "weight")):
        raise ValueError(f"{context} must contain latitude, longitude, and weight.")
    if out.duplicated([lat, lon, "target_id"], keep=False).any():
        raise ValueError(f"{context} contains duplicate latitude/longitude/target_id rows.")
    return out


class SparseTransform:
    """Target-by-source weights (CSR), for weighted means over each target's
    observed sources."""

    def __init__(self, weighted_matrix, source_ids: tuple, target_ids: tuple):
        self.weighted_matrix = weighted_matrix
        self.source_ids, self.target_ids = source_ids, target_ids
        self.source_index = {source: index for index, source in enumerate(source_ids)}

    @classmethod
    def compile(cls, weights: pd.DataFrame, *, source_ids=None, target_ids=None, source_col="source_id",
                target_col="target_id", weight_col="weight") -> "SparseTransform":
        """From a long (source, target, weight) table."""
        missing = {source_col, target_col, weight_col} - set(weights.columns)
        if missing:
            raise ValueError(f"Sparse cell transform is missing columns: {sorted(missing)}")
        if weights.empty:
            raise ValueError("Sparse cell-transform weights must not be empty.")
        values = pd.to_numeric(weights[weight_col], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Compiled cell-transform weights must be finite and non-negative.")
        if (pd.Series(values).groupby(weights[target_col].to_numpy()).sum() <= 0).any():
            raise ValueError("Every compiled cell-transform target must have positive weight.")
        positive = values > 0
        rows = weights.loc[positive]
        source_ids = tuple(pd.unique(rows[source_col]) if source_ids is None else source_ids)
        target_ids = tuple(pd.unique(rows[target_col]) if target_ids is None else target_ids)
        source_pos = rows[source_col].map({s: i for i, s in enumerate(source_ids)})
        target_pos = rows[target_col].map({t: i for i, t in enumerate(target_ids)})
        if source_pos.isna().any() or target_pos.isna().any():
            raise ValueError("Cell-transform weights contain IDs outside the compiled axes.")
        matrix = sparse.csr_matrix(
            (values[positive], (target_pos.to_numpy(int), source_pos.to_numpy(int))),
            shape=(len(target_ids), len(source_ids)),
        )
        return cls(matrix, source_ids, target_ids)

    def supported(self, present: np.ndarray) -> np.ndarray:
        """(target, column): whether any of a target's sources is present
        (present: a (source, column) mask)."""
        support = self.weighted_matrix.copy()
        support.data = np.ones_like(support.data)
        return np.asarray(support @ np.asarray(present, dtype=float)) > 0

    def observed_mean(self, source_values: np.ndarray, column_chunk_size: int = 256) -> np.ndarray:
        """W @ filled / W @ observed for a (source, column) matrix, NaN where a
        target has no observed source."""
        source_values = np.asarray(source_values, dtype=float)
        if source_values.ndim != 2 or source_values.shape[0] != len(self.source_ids):
            raise ValueError("Sparse transform source axis does not match the compiled weight matrix.")
        out = np.full((len(self.target_ids), source_values.shape[1]), np.nan)
        for start in range(0, source_values.shape[1], int(column_chunk_size)):
            stop = min(source_values.shape[1], start + int(column_chunk_size))
            block = source_values[:, start:stop]
            valid = np.isfinite(block)
            numerator = np.asarray(self.weighted_matrix @ np.where(valid, block, 0.0))
            denominator = np.asarray(self.weighted_matrix @ valid.astype(float))
            np.divide(numerator, denominator, out=out[:, start:stop], where=denominator > 0)
        return out


def point_sources(points: pd.DataFrame, transform) -> np.ndarray:
    """Each point's position among the transform's sources (-1 for none)."""
    index = transform.source_index
    return np.array(
        [index.get((float(a), float(b)), -1) for a, b in zip(points["lat"], points["lon"])], dtype=np.int64
    )


def grid_axes(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The distinct latitudes and longitudes of a grid's points, in file order."""
    return pd.unique(points["lat"].to_numpy()), pd.unique(points["lon"].to_numpy())
