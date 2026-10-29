"""NetCDF input: a config's files, each file's format, and its rainfall as arrays.

Files: the files a FileSet names, in year order.

Formats: the files of one model need not be laid out alike. The archives a
blend trains on and the live files it forecasts from may name their
dimensions differently and store rainfall in other units, so each file's
format is resolved when it is read, by the rules ai-almanac registers
datasets by:

- a dimension's role (time, day, number, lat, lon, adm3_name) comes from
  the model's configured `dims`, else its name (case-insensitive aliases),
  else its coordinate (a lead dimension counted in days);
- rainfall is converted to millimetres from the variable's declared
  `units`; the model's `unit_scale` applies only to files declaring none.

Everything downstream, the forecast signature included, sees the resolved
dimension names and millimetres.

Reading: values come out as the legacy table readers produce them (same
dtype, NaN for masked values, the same scaling), but as arrays over the
file's dimensions rather than one table row per value, and a forecast is
read only in the storage chunks that hold a kept point.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .._legacy import nc_utils
from ..config import BlendConfig, FileSet

# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

_YEAR = re.compile(r"(19|20)\d{2}")


def file_year(path: str | Path) -> int | None:
    """The season a file holds: the first 19xx/20xx in its name."""
    match = _YEAR.search(Path(path).name)
    return int(match.group()) if match else None


def list_files(files: FileSet, config: BlendConfig, what: str) -> list[Path]:
    """The files a FileSet names, in year order."""
    if files.paths:
        paths = [config.path(p) for p in files.paths]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{what}: missing files: {', '.join(missing)}")
    else:
        folder = config.path(files.folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"{what}: {folder} is not a directory")
        pattern = re.compile(files.pattern)
        paths = [p for p in folder.iterdir() if p.is_file() and pattern.search(p.name)]
    if files.years:
        wanted = set(files.years)
        paths = [p for p in paths if file_year(p) in wanted]
    if not paths:
        raise FileNotFoundError(f"{what}: no files match {files}")
    return sorted(paths, key=lambda p: (file_year(p) or 0, str(p)))


# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------

# Millimetres per unit of a rainfall total's declared (CF) units.
MILLIMETRES_PER_UNIT = {
    "m": 1000.0,
    "metres": 1000.0,
    "meters": 1000.0,
    "m/day": 1000.0,
    "mm": 1.0,
    "mm/day": 1.0,
    "mm day-1": 1.0,
    "mm d-1": 1.0,
    "kg m-2": 1.0,
    "kg m**-2": 1.0,
    "kg/m2": 1.0,
}

# Dimension names by role, lower case.
_ALIASES = {
    "time": "time", "init_time": "time", "initial_time": "time", "forecast_reference_time": "time",
    "day": "day", "lead_day": "day", "lead_days": "day", "prediction_timedelta_daily": "day",
    "number": "number", "member": "number", "members": "number", "ensemble": "number",
    "ensemble_member": "number", "realization": "number", "sample": "number",
    "lat": "lat", "latitude": "lat",
    "lon": "lon", "longitude": "lon",
    "adm3": "adm3_name", "adm3_name": "adm3_name",
}
_DAY_UNITS = ("day", "days")


@dataclass(frozen=True)
class ForecastFormat:
    """How one file is read: its dimensions' roles and its rainfall scale."""

    rename: dict[str, str]   # file dimension -> role, for every dimension with one
    scale: float             # multiplies the file's rainfall into millimetres
    declared_units: str | None

    @property
    def units(self) -> str:
        """The rainfall units the blend reads: millimetres, or how they were taken."""
        return "mm" if self.declared_units is not None else f"undeclared (x{self.scale:g})"


def forecast_format(ds, variable: str, dims: Mapping[str, str], unit_scale: float, where="") -> ForecastFormat:
    """The format of an open netCDF4 Dataset's forecast variable."""
    if variable not in ds.variables:
        raise ValueError(f"{where}: no {variable!r} variable")
    var = ds.variables[variable]
    configured = {str(k).lower(): v for k, v in dims.items()}
    rename: dict[str, str] = {}
    for dim in var.dimensions:
        role = configured.get(dim.lower()) or _ALIASES.get(dim.lower())
        if role is None and dim in ds.variables:
            units = str(getattr(ds.variables[dim], "units", "")).strip().lower()
            if units in _DAY_UNITS:
                role = "day"
        if role is not None:
            rename[dim] = role
    roles = list(rename.values())
    repeated = sorted({role for role in roles if roles.count(role) > 1})
    if repeated:
        raise ValueError(f"{where}: several dimensions of {variable!r} read as {', '.join(repeated)}")
    declared = getattr(var, "units", None)
    declared = str(declared).strip() if declared is not None and str(declared).strip() else None
    return ForecastFormat(rename=rename, scale=rainfall_scale(declared, unit_scale, where), declared_units=declared)


def rainfall_scale(units: str | None, fallback: float, where="") -> float:
    """Millimetres per unit of rainfall declared in `units`; fallback when undeclared."""
    if units is None:
        return float(fallback)
    try:
        return MILLIMETRES_PER_UNIT[units.lower()]
    except KeyError:
        supported = ", ".join(repr(u) for u in MILLIMETRES_PER_UNIT)
        raise ValueError(f"{where}: unrecognised rainfall units {units!r} (supported: {supported})") from None


def of_file(path, variable: str, dims: Mapping[str, str], unit_scale: float) -> ForecastFormat:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        return forecast_format(ds, variable, dims, unit_scale, where=str(path))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def lead_days_in_order(positions, day_numbers, where) -> tuple[np.ndarray, list[int]]:
    """The selected lead days ascending, with their positions in the file.
    Onsets are found over consecutive days and binned by lead day, so the
    days must be whole, distinct and without gaps, whatever the file's order."""
    if any(d is None for d in day_numbers):
        raise ValueError(f"{where}: lead days must be whole numbers of days")
    order = np.argsort(np.asarray(day_numbers), kind="stable")
    days = [int(day_numbers[i]) for i in order]
    if len(set(days)) != len(days) or any(b - a != 1 for a, b in zip(days, days[1:])):
        raise ValueError(f"{where}: lead days must be consecutive and distinct, got {days[:8]}{'…' if len(days) > 8 else ''}")
    return np.asarray(positions)[order], days


def _ranges(chunk: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(size, start + chunk)) for start in range(0, size, chunk)]


class ForecastFile:
    """A forecast file's rainfall at the kept points, read a run of issues at
    a time (whole storage chunks, each decompressed once, and only where a
    kept point lies or, until member spread is found, a spread-area point).

        with ForecastFile(path, variable, rename, spec, keep) as f:
            for start, stop, rain in f.issue_runs():   # rain: (issue, member, point, day)
                ...
    """

    def __init__(self, path, variable: str, rename: dict, spec: dict, keep: np.ndarray, *,
                 spread_over: np.ndarray | None = None, scale: float = 1.0):
        import netCDF4

        self.path, self.scale = path, scale
        self._ds = netCDF4.Dataset(path)
        try:
            self._setup(variable, rename, spec, keep, spread_over)
        except BaseException:
            self._ds.close()
            raise

    def _setup(self, variable, rename, spec, keep, spread_over):
        ds = self._ds
        if variable not in ds.variables:
            raise ValueError(f"{self.path}: no {variable!r} variable")
        layout = nc_utils._forecast_layout(ds, variable, rename, "day")
        if layout is None:
            raise ValueError(f"{self.path}: a layout the point reader does not support")
        self._v = v = ds.variables[variable]
        dim_names, dim_vals = layout["dim_names"], layout["dim_vals"]
        day_idx, other_idx, n_leading = layout["day_idx"], layout["other_idx"], layout["n_leading"]
        positions, _, day_numbers = nc_utils._select_forecast_days(dim_vals[dim_names[day_idx]], spec, self.path)
        self._day_positions, self.days = lead_days_in_order(positions, day_numbers, self.path)
        spatial = [dim_names[i] for i in other_idx[n_leading:]]
        shape = [len(dim_vals[d]) for d in spatial]
        n_points = int(np.prod(shape))
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (n_points,):
            raise ValueError(f"keep must have one entry per point ({n_points}).")
        self._kept = kept = np.flatnonzero(keep)
        self._position = np.full(n_points, -1, dtype=np.int64)
        self._position[kept] = np.arange(len(kept))
        self.points = nc_utils._named_grid(
            nc_utils._product_frame(spatial, [dim_vals[d] for d in spatial]), rename, layout["adm3_dim_name"]
        ).take(kept).reset_index(drop=True)

        self._n_times = len(dim_vals[dim_names[0]])
        self._member_dim = other_idx[1] if layout["has_members"] else None
        self._n_members = len(dim_vals[dim_names[self._member_dim]]) if self._member_dim is not None else 1
        self.members = (
            np.asarray(dim_vals[dim_names[self._member_dim]]).astype(int) if self._member_dim is not None else None
        )
        self._spread_area = np.zeros(n_points, dtype=bool)
        if spread_over is not None and self._member_dim is not None:
            self._spread_area[np.asarray(spread_over, dtype=bool)] = True
        self.member_spread = False
        self._present = None  # members present per (time, point, day), when chunks split the members

        chunks = v.chunking()
        self._spatial_axes = [dim_names.index(d) for d in spatial]
        tile_sizes = [int(chunks[a]) for a in self._spatial_axes] if isinstance(chunks, list) else shape
        grid_index = np.arange(n_points).reshape(shape)
        self._tiles = []
        for corner in np.ndindex(*[len(_ranges(t, size)) for t, size in zip(tile_sizes, shape)]):
            bounds = [_ranges(t, size)[c] for t, size, c in zip(tile_sizes, shape, corner)]
            self._tiles.append((bounds, grid_index[tuple(slice(a, b) for a, b in bounds)].reshape(-1)))
        self._perm = other_idx + [day_idx]
        self._dim_names = dim_names
        # The issue dimension's own units, whatever it is called.
        time_dim = dim_names[0]
        units = getattr(ds.variables[time_dim], "units", None) if time_dim in ds.variables else None
        units = units or nc_utils.get_nc_time(ds)["units"]
        stamps = pd.DatetimeIndex(nc_utils.nc_time_to_dates(dim_vals[time_dim], units))
        self.issues = stamps.normalize().to_numpy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._ds.close()

    def issue_runs(self):
        """(start, stop, rain) for consecutive runs of issues; rain holds every
        member: (issue, member, point, day), one member when deterministic."""
        runs = {}
        for t0, t1, m0, m1 in nc_utils._leading_tiles(self._v, self._n_times, self._n_members, self._member_dim):
            runs.setdefault((t0, t1), []).append((m0, m1))
        for (t0, t1), member_ranges in runs.items():
            rain = None
            for m0, m1 in member_ranges:
                rain = self._read(t0, t1, m0, m1, rain)
            if rain is None:
                rain = np.empty((t1 - t0, self._n_members, 0, len(self.days)), dtype=np.float32)
            yield t0, t1, rain

    def _read(self, t0, t1, m0, m1, rain):
        for bounds, cells in self._tiles:
            kept_here = self._position[cells] >= 0
            area_here = self._spread_area[cells] if not self.member_spread else np.zeros(len(cells), dtype=bool)
            if not kept_here.any() and not area_here.any():
                continue
            index = [slice(None)] * len(self._dim_names)
            index[0] = slice(t0, t1)
            if self._member_dim is not None:
                index[self._member_dim] = slice(m0, m1)
            for axis, (a, b) in zip(self._spatial_axes, bounds):
                index[axis] = slice(a, b)
            block = nc_utils._filled_with_nan(self._v[tuple(index)])
            block = np.transpose(block, self._perm)[..., self._day_positions]
            block = block.reshape(t1 - t0, m1 - m0, len(cells), len(self.days))
            if area_here.any():
                counts = (~np.isnan(block[:, :, area_here, :])).sum(axis=1, dtype=np.int32)
                if m1 - m0 < self._n_members:
                    if self._present is None:
                        self._present = np.zeros((self._n_times, len(self._spread_area), len(self.days)), dtype=np.int32)
                    self._present[t0:t1, cells[area_here]] += counts
                    counts = self._present[t0:t1, cells[area_here]]
                self.member_spread = bool((counts >= 2).any())
            if kept_here.any():
                part = block[:, :, kept_here, :]
                if self.scale != 1.0:
                    part = part * self.scale
                if rain is None:
                    rain = np.empty((t1 - t0, self._n_members, len(self._kept), len(self.days)), dtype=part.dtype)
                rain[:, m0:m1, self._position[cells[kept_here]], :] = part
        return rain


@dataclass
class ObservationArray:
    """An observation file's rainfall, all points."""

    points: pd.DataFrame   # named as the legacy reader names them (lat/lon, or id)
    dates: np.ndarray      # datetime64[ns], one per time step
    years: np.ndarray      # the year of each time step
    rain: np.ndarray       # (time, point), the file's dtype, NaN where masked


def read_observations(path, variable: str, rename: dict, *, shift_days: int = 0) -> ObservationArray:
    """The rainfall of an observation file, as nc_read_groundtruth_long reads it
    but as a (time, point) array; the calendar must be daily without gaps.

    shift_days moves the dates onto the forecasts' labels (1 for files whose
    date is the day the rain fell; see Observations.date_marks); each day keeps
    the season of the calendar day it covers."""
    import os

    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if variable not in ds.variables:
            found = ", ".join(ds.variables)
            raise ValueError(f"{os.path.basename(path)} has no {variable!r} variable (it has: {found})")
        v = ds.variables[variable]
        dim_names = list(v.dimensions)
        dim_vals, adm3_dim = {}, None
        for d in dim_names:
            if d.lower() in ("adm3_name", "adm3"):
                adm3_dim = d
                dim_vals[d] = nc_utils.get_nc_adm3_names(ds)
            elif d.lower() == "time":
                dim_vals[d] = nc_utils.get_nc_time(ds)["values"]
            elif d in ds.variables:
                dim_vals[d] = np.array(ds[d][:]).flatten()
            else:
                dim_vals[d] = np.arange(ds.dimensions[d].size)
        time_dims = [d for d in dim_names if d.lower() == "time"]
        if len(time_dims) != 1:
            raise ValueError(f"{os.path.basename(path)}: expected one time dimension in {dim_names}")
        time_dim = time_dims[0]
        others = [d for d in dim_names if d != time_dim]
        rain = nc_utils._filled_with_nan(v[:])
        rain = np.moveaxis(rain, dim_names.index(time_dim), 0).reshape(len(dim_vals[time_dim]), -1)
        units = nc_utils.get_nc_time(ds)["units"]
    stamps = pd.DatetimeIndex(nc_utils.nc_time_to_dates(dim_vals[time_dim], units))
    calendar = pd.Series(stamps).sort_values().reset_index(drop=True)
    name = os.path.basename(path)
    if calendar.isna().any() or calendar.duplicated().any():
        raise ValueError(f"{name}: ground-truth time coordinate must contain one valid date per day.")
    steps = calendar.groupby(calendar.dt.year).diff().dt.days
    gaps = steps.notna() & steps.ne(1)
    if gaps.any():
        at = int(np.flatnonzero(gaps.to_numpy())[0])
        raise ValueError(
            f"{name}: ground-truth time coordinate is not daily-contiguous; first gap is "
            f"{calendar.iloc[at - 1].date()} to {calendar.iloc[at].date()}."
        )
    points = nc_utils._named_grid(
        nc_utils._product_frame(others, [dim_vals[d] for d in others]), rename, adm3_dim
    )
    dates = (stamps.normalize() + pd.Timedelta(days=int(shift_days))).to_numpy()
    return ObservationArray(points=points, dates=dates, years=stamps.year.to_numpy(), rain=rain)
