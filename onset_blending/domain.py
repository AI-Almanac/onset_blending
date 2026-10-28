"""Where a blend applies: unit ids, the area of interest, dissemination units,
and regridding inputs onto the units."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import _parallel
from ._legacy import nc_utils
from ._legacy.spatial_id_utils import GridIdConvention, ensure_spatial_id_col, format_coord_component
from .config import BlendConfig

# Peak bytes per value of the largest variable while one file is regridded.
_REGRID_BYTES_PER_VALUE = 18
# The regridder reads the grid's coordinates under these names.
_COORDINATE_NAMES = {"lat": "lat", "latitude": "lat", "lon": "lon", "longitude": "lon"}


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


def _has_unit_dimension(path: Path) -> bool:
    import xarray as xr

    with xr.open_dataset(path, decode_times=False) as ds:
        names = {str(name).lower() for name in [*ds.dims, *ds.coords]}
    return bool(names & {"adm3", "adm3_name"})


def _largest_value_count(path: Path) -> int:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return max(
            (math.prod(var.shape) for var in ds.variables.values() if len(var.dimensions) >= 2),
            default=0,
        )


def _regrid_file(src: Path, context: dict) -> Path:
    """One file aggregated onto the units, written into context["out_dir"]."""
    import xarray as xr

    from ._legacy.remap_nc import batch_aggregate_to_adm3_matrix

    work = Path(tempfile.mkdtemp(dir=context["work_dir"]))
    staged = work / src.name
    with xr.open_dataset(src, mask_and_scale=True) as ds:
        rename = {
            name: _COORDINATE_NAMES[str(name).lower()]
            for name in [*ds.dims, *ds.coords]
            if _COORDINATE_NAMES.get(str(name).lower(), name) != name
        }
        if rename:
            ds.rename(rename).to_netcdf(staged)
        else:
            staged.symlink_to(src.resolve())
    batch_aggregate_to_adm3_matrix(str(work), context["weights"], input_file=str(staged))
    regridded = staged.with_name(f"{staged.stem}_adm3{staged.suffix}")
    if not regridded.is_file():
        raise FileNotFoundError(f"Regridding {src.name} produced no {regridded.name}")
    target = Path(context["out_dir"]) / src.name
    shutil.move(str(regridded), target)
    shutil.rmtree(work, ignore_errors=True)
    return target


@contextlib.contextmanager
def on_units(paths: list[Path], weights: Path | None, *, workers: int = 1, budget_bytes: int | None = None):
    """The inputs as files on the blend's units: regridded with `weights` into
    a temporary directory that lasts while the block runs, or the files
    themselves when there is nothing to regrid."""
    pending = [p for p in paths if weights is not None and not _has_unit_dimension(p)]
    if not pending:
        yield list(paths)
        return
    with tempfile.TemporaryDirectory(prefix="onset-blending-units-") as tmp:
        out_dir = Path(tmp) / "units"
        out_dir.mkdir()
        context = {"weights": str(weights), "out_dir": str(out_dir), "work_dir": tmp}
        count = _parallel.workers_within(
            [_largest_value_count(p) * _REGRID_BYTES_PER_VALUE for p in pending], workers, budget_bytes
        )
        regridded = dict(zip(pending, _parallel.map_ordered(_regrid_file, pending, context, count)))
        yield [regridded.get(p, p) for p in paths]
