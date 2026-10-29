"""Where a blend applies: unit ids, the area of interest, dissemination units,
and regridding inputs onto the units."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ._legacy import nc_utils
from ._legacy.spatial_id_utils import GridIdConvention, ensure_spatial_id_col, format_coord_component
from .config import BlendConfig



def id_convention(config: BlendConfig) -> GridIdConvention | None:
    """The configured "<lat>_<lon>" convention; None means inferred per table."""
    ids = config.domain.grid_ids
    if ids.decimal_digits is None:
        return None
    return GridIdConvention(decimal_digits=ids.decimal_digits, number_format=ids.format, source="spec")


def with_ids(df: pd.DataFrame, convention: GridIdConvention | None) -> pd.DataFrame:
    """df with an `id` column: kept if present, else "<lat>_<lon>" from each
    row's coordinates. Each distinct coordinate is formatted once; a grid has
    thousands of cells where its rows run to tens of millions."""
    if "id" in df.columns:
        return df
    if convention is None or not {"lat", "lon"} <= set(df.columns):
        return ensure_spatial_id_col(df, context="input spatial IDs")
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
        self._filter: dict = {}
        if domain.centroids is not None:
            self._filter["centroids_file"] = str(config.path(domain.centroids.file))
            if domain.centroids.id_col:
                self._filter["centroids_id_col"] = domain.centroids.id_col
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
            self.dissemination = {"dissemination_cells_file": str(config.path(domain.dissemination.file))}
            if domain.dissemination.id_col:
                self.dissemination["id_col"] = domain.dissemination.id_col
        ids = config.domain.grid_ids
        self._geometry = (
            {"geometry": {"grid_id_decimal_digits": ids.decimal_digits, "grid_id_format": ids.format}}
            if ids.decimal_digits is not None
            else {}
        )

    @classmethod
    def everything(cls) -> "AreaFilter":
        """No area of interest and no dissemination filter (a trained blend's
        units are given instead)."""
        area = cls.__new__(cls)
        area._filter, area.bbox, area.outlines, area.dissemination, area._geometry = {}, None, None, None, {}
        return area

    @property
    def active(self) -> bool:
        return bool(self.bbox or self.outlines or self.dissemination)

    def area(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows inside the area of interest."""
        if self.bbox:
            kept = nc_utils.filter_by_dissemination_cells(
                df, {"filter": {"bbox": self.bbox, **self._filter}, **self._geometry}
            )
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
            df = nc_utils.filter_by_dissemination_cells(
                df, {"filter": self.dissemination, **self._geometry}
            )
        return df

    def _inside_outlines(self, df: pd.DataFrame) -> pd.DataFrame:
        import shapely
        from shapely.geometry import shape

        lat, lon = nc_utils._resolve_unit_latlon(df, self._filter)
        if lat is None:
            raise ValueError("The area of interest needs a lat/lon per row, or domain.centroids")
        outline = shapely.union_all([shape(f["geometry"]) for f in self.outlines["features"]])
        return df[shapely.contains_xy(outline, lon, lat)]


class Regridder:
    """Area-weighted means of gridded values over the blend's units: the
    sparse transform the legacy regrid applied file by file (a weighted mean
    over each unit's observed source cells), applied to arrays as read."""

    def __init__(self, weights: pd.DataFrame):
        from ._legacy.geometry_utils import normalize_regrid_weights

        mapping = normalize_regrid_weights(weights, context="regrid weights")
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
        from ._legacy.sparse_transform_utils import compile_sparse_cell_transform

        sources = tuple((float(a), float(b)) for a in lats for b in lons)
        if sources not in self._transforms:
            self._transforms[sources] = compile_sparse_cell_transform(
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
        from ._legacy.sparse_transform_utils import sparse_observed_weighted_mean

        return sparse_observed_weighted_mean(transform, values)


def point_sources(points: pd.DataFrame, transform) -> np.ndarray:
    """Each point's position among the transform's sources (-1 for none)."""
    index = transform.source_index
    return np.array(
        [index.get((float(a), float(b)), -1) for a, b in zip(points["lat"], points["lon"])], dtype=np.int64
    )


def grid_axes(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """The distinct latitudes and longitudes of a grid's points, in file order."""
    return pd.unique(points["lat"].to_numpy()), pd.unique(points["lon"].to_numpy())
