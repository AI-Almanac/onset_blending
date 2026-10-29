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

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

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

        with ForecastFile(path, variable, rename, leads, keep) as f:
            for start, stop, rain in f.issue_runs():   # rain: (issue, member, point, day)
                ...
    """

    def __init__(self, path, variable: str, rename: dict, leads: LeadSelection, keep: np.ndarray, *,
                 spread_over: np.ndarray | None = None, scale: float = 1.0):
        import netCDF4

        self.path, self.scale = path, scale
        self._ds = netCDF4.Dataset(path)
        try:
            self._setup(variable, rename, leads, keep, spread_over)
        except BaseException:
            self._ds.close()
            raise

    def _setup(self, variable, rename, leads, keep, spread_over):
        ds = self._ds
        if variable not in ds.variables:
            raise ValueError(f"{self.path}: no {variable!r} variable")
        layout = _forecast_layout(ds, variable, rename, "day")
        if layout is None:
            raise ValueError(f"{self.path}: a layout the point reader does not support")
        self._v = v = ds.variables[variable]
        dim_names, dim_vals = layout["dim_names"], layout["dim_vals"]
        day_idx, time_idx, member_idx = layout["day_idx"], layout["time_idx"], layout["member_idx"]
        positions, _, day_numbers = _select_forecast_days(dim_vals[dim_names[day_idx]], leads, self.path)
        self._day_positions, self.days = lead_days_in_order(positions, day_numbers, self.path)
        spatial = [dim_names[i] for i in layout["spatial_idx"]]
        shape = [len(dim_vals[d]) for d in spatial]
        n_points = int(np.prod(shape))
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (n_points,):
            raise ValueError(f"keep must have one entry per point ({n_points}).")
        self._kept = kept = np.flatnonzero(keep)
        self._position = np.full(n_points, -1, dtype=np.int64)
        self._position[kept] = np.arange(len(kept))
        self.points = _named_grid(
            _product_frame(spatial, [dim_vals[d] for d in spatial]), rename, layout["adm3_dim_name"]
        ).take(kept).reset_index(drop=True)

        self._time_dim = time_idx
        self._n_times = len(dim_vals[dim_names[time_idx]])
        self._member_dim = member_idx
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
        # Read blocks are put in (time, member, points..., day) order.
        self._perm = [time_idx] + ([member_idx] if member_idx is not None else []) + layout["spatial_idx"] + [day_idx]
        self._dim_names = dim_names
        # The issue dimension's own units, whatever it is called.
        time_dim = dim_names[time_idx]
        units = getattr(ds.variables[time_dim], "units", None) if time_dim in ds.variables else None
        units = units or nc_time(ds)["units"]
        stamps = pd.DatetimeIndex(time_to_dates(dim_vals[time_dim], units))
        self.issues = stamps.normalize().to_numpy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._ds.close()

    def issue_runs(self):
        """(start, stop, rain) for consecutive runs of issues; rain holds every
        member: (issue, member, point, day), one member when deterministic."""
        runs = {}
        for t0, t1, m0, m1 in _leading_tiles(self._v, self._n_times, self._n_members, self._member_dim, self._time_dim):
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
            index[self._time_dim] = slice(t0, t1)
            if self._member_dim is not None:
                index[self._member_dim] = slice(m0, m1)
            for axis, (a, b) in zip(self._spatial_axes, bounds):
                index[axis] = slice(a, b)
            block = _filled_with_nan(self._v[tuple(index)])
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
                dim_vals[d] = adm3_names(ds)
            elif d.lower() == "time":
                dim_vals[d] = nc_time(ds)["values"]
            elif d in ds.variables:
                dim_vals[d] = np.array(ds[d][:]).flatten()
            else:
                dim_vals[d] = np.arange(ds.dimensions[d].size)
        time_dims = [d for d in dim_names if d.lower() == "time"]
        if len(time_dims) != 1:
            raise ValueError(f"{os.path.basename(path)}: expected one time dimension in {dim_names}")
        time_dim = time_dims[0]
        others = [d for d in dim_names if d != time_dim]
        rain = _filled_with_nan(v[:])
        rain = np.moveaxis(rain, dim_names.index(time_dim), 0).reshape(len(dim_vals[time_dim]), -1)
        units = nc_time(ds)["units"]
    stamps = pd.DatetimeIndex(time_to_dates(dim_vals[time_dim], units))
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
    points = _named_grid(
        _product_frame(others, [dim_vals[d] for d in others]), rename, adm3_dim
    )
    dates = (stamps.normalize() + pd.Timedelta(days=int(shift_days))).to_numpy()
    return ObservationArray(points=points, dates=dates, years=stamps.year.to_numpy(), rain=rain)


# ---------------------------------------------------------------------------
# NetCDF primitives: coordinates, times, layouts, reading in storage chunks
# ---------------------------------------------------------------------------

# Dimension names the readers know, mapped to theirs; a source's own `dims` win.
OBSERVATION_DIMS = {
    "TIME": "time", "time": "time",
    "LATITUDE": "lat", "latitude": "lat", "lat": "lat",
    "LONGITUDE": "lon", "longitude": "lon", "lon": "lon",
    "ADM3_NAME": "adm3_name", "adm3_name": "adm3_name", "adm3": "adm3_name",
}
FORECAST_DIMS = {
    "time": "time", "day": "day", "lat": "lat", "lon": "lon",
    "adm3_name": "adm3_name", "adm3": "adm3_name", "number": "number", "sample": "number",
}


@dataclass(frozen=True)
class LeadSelection:
    """The lead days a model reads (min..max), and its strict rain horizon:
    with rain_day_max the files must hold days 1..rain_day_max (exactly, or
    at least under the truncate policy)."""

    min_day: int | None = None
    max_day: int | None = None
    rain_day_max: int | None = None
    rain_horizon_policy: str | None = None

    @classmethod
    def of(cls, model) -> LeadSelection:
        horizon = model.rain_horizon
        return cls(
            min_day=model.lead_days.min,
            max_day=model.lead_days.max,
            rain_day_max=horizon.max_day,
            rain_horizon_policy=horizon.resolved_policy if horizon.max_day is not None else None,
        )

_SPATIAL_DIMS = ("lat", "lon", "id")
# Values per slab read at once (~64 MB as float32): a slab of whole storage
# chunks along the leading dimension, so each chunk is decompressed once.
_READ_SLAB_VALUES = 1 << 24


def adm3_names(ds) -> list[str]:
    """The adm3_name string coordinate of a netCDF4 Dataset (any common spelling)."""
    import netCDF4

    for name in ("adm3_name", "ADM3_NAME", "Adm3_Name", "adm3"):
        if name in ds.variables:
            var = ds.variables[name]
            try:
                items = var[:].tolist()
            except (NotImplementedError, TypeError):
                items = netCDF4.chartostring(var[:]).tolist()  # a fixed-length char dimension
            return [item.decode("utf-8").strip() if isinstance(item, (bytes, bytearray)) else str(item).strip()
                    for item in items]
    raise ValueError(
        "Could not find adm3_name variable/dimension in NetCDF file. "
        f"Available variables: {list(ds.variables.keys())}"
    )


def nc_time(ds) -> dict:
    """The time coordinate of a netCDF4 Dataset: values, units, name, source."""
    names = ["TIME", "time", "Time", "t", "T"]
    for name in names:
        if name in ds.variables:
            return {"values": np.array(ds[name][:]).flatten(), "units": getattr(ds[name], "units", None),
                    "name": name, "source": "var"}
    for name in names:
        for variable in ds.variables:
            if variable.lower() == name.lower():
                return {"values": np.array(ds[variable][:]).flatten(), "units": getattr(ds[variable], "units", None),
                        "name": variable, "source": "var_ci"}
    for name in names:
        if name in ds.dimensions:
            return {"values": np.arange(ds.dimensions[name].size, dtype=float), "units": None, "name": name,
                    "source": "dim"}
    raise ValueError("Could not find time variable/dimension in NetCDF file")


def time_to_dates(values, units) -> pd.DatetimeIndex:
    """Numeric NetCDF times as timestamps, from CF units ("<unit>s since <origin>")."""
    match = re.match(r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+)\s*$", str(units), re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized NetCDF time units: {units}")
    unit = match.group(1).lower().rstrip("s")
    origin_text = match.group(2).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            origin = datetime.strptime(origin_text, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Could not parse NetCDF time origin: {origin_text}")
    seconds = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
    return pd.DatetimeIndex([origin + timedelta(seconds=float(t) * seconds) for t in values])


def _filled_with_nan(raw) -> np.ndarray:
    """np.ma.filled(raw, NaN) for an array just read from a file, filled in
    place rather than copied when the array is a float buffer it owns."""
    if not np.ma.isMaskedArray(raw):
        return np.asarray(raw)
    data = raw.data
    if not np.issubdtype(data.dtype, np.floating) or not data.flags.writeable or not data.flags.owndata:
        return np.ma.filled(raw, np.nan)
    if raw.mask is not np.ma.nomask:
        data[np.ma.getmaskarray(raw)] = np.nan
    return data


def _product_frame(names, values) -> pd.DataFrame:
    """The table of every combination of the dimensions' values (the first
    varying slowest), each column with the dtype pandas infers for it."""
    sizes = [len(v) for v in values]
    if not sizes or 0 in sizes:
        from itertools import product

        return pd.DataFrame(list(product(*values)), columns=names)
    n_rows = int(np.prod(sizes))
    columns = {}
    for i, (name, vals) in enumerate(zip(names, values)):
        inner = int(np.prod(sizes[i + 1:]))
        outer = n_rows // (sizes[i] * inner)
        column = pd.DataFrame([(v,) for v in vals]).iloc[:, 0]
        columns[name] = column.take(np.tile(np.repeat(np.arange(sizes[i]), inner), outer)).reset_index(drop=True)
    return pd.DataFrame(columns)


def _renamed(frame: pd.DataFrame, rename: Mapping[str, str] | None) -> pd.DataFrame:
    """Columns renamed old -> new, matching old names case-insensitively."""
    if not rename:
        return frame
    mapping = {}
    for old, new in rename.items():
        for column in frame.columns:
            if column.lower() == old.lower():
                mapping[column] = new
                break
    return frame.rename(columns=mapping)


def _named_grid(frame: pd.DataFrame, rename, adm3_dim_name) -> pd.DataFrame:
    """A grid table of raw dimension columns, renamed: lower case, and the
    administrative units as `id`."""
    frame = _renamed(frame, rename or {})
    frame.columns = frame.columns.str.lower()
    adm3 = adm3_dim_name.lower() if adm3_dim_name else None
    if adm3 and adm3 in frame.columns:
        frame = frame.rename(columns={adm3: "id"})
    if "adm3_name" in frame.columns:
        frame = frame.rename(columns={"adm3_name": "id"})
    return frame


def _forecast_layout(ds, variable: str, rename, day_dim: str) -> dict | None:
    """The forecast variable's layout for point-wise reading, or None: the
    issue time, the lead day, optionally the member, and spatial dimensions
    (lat/lon, or adm3_name read as `id`), in any order and nothing else."""
    v = ds.variables[variable]
    dim_names = list(v.dimensions)
    dim_vals = {}
    adm3_dim_name = None
    for d in dim_names:
        lower = d.lower()
        if lower in ("adm3_name", "adm3"):
            adm3_dim_name = d
            dim_vals[d] = adm3_names(ds)
        elif lower == "time":
            dim_vals[d] = nc_time(ds)["values"]
        elif d in ds.variables:
            dim_vals[d] = np.array(ds[d][:]).flatten()
        elif d in ds.dimensions:
            dim_vals[d] = np.arange(ds.dimensions[d].size)
        else:
            dim_vals[d] = np.arange(len(v))
    renamed = list(_renamed(pd.DataFrame({d: [dim_vals[d][0]] for d in dim_names}), rename or {}).columns)
    renamed = [c.lower() for c in renamed]
    adm3 = adm3_dim_name.lower() if adm3_dim_name else None
    renamed = ["id" if name in ("adm3_name", adm3) else name for name in renamed]
    roles = {"time": [], "number": [], day_dim.lower(): [], "spatial": []}
    for i, name in enumerate(renamed):
        if name in _SPATIAL_DIMS:
            roles["spatial"].append(i)
        elif name in roles:
            roles[name].append(i)
        else:
            return None
    if len(roles["time"]) != 1 or len(roles[day_dim.lower()]) != 1 or len(roles["number"]) > 1 or not roles["spatial"]:
        return None
    return {
        "dim_names": dim_names,
        "dim_vals": dim_vals,
        "adm3_dim_name": adm3_dim_name,
        "day_idx": roles[day_dim.lower()][0],
        "time_idx": roles["time"][0],
        "member_idx": roles["number"][0] if roles["number"] else None,
        "spatial_idx": roles["spatial"],
    }


def forecast_points(path, variable: str, rename) -> pd.DataFrame | None:
    """The forecast variable's spatial points, in file order (lat/lon, or id);
    None without the variable or for a layout the reader does not support."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if variable not in ds.variables:
            return None
        layout = _forecast_layout(ds, variable, rename, "day")
        if layout is None:
            return None
        spatial = [layout["dim_names"][i] for i in layout["spatial_idx"]]
        return _named_grid(_product_frame(spatial, [layout["dim_vals"][d] for d in spatial]), rename, layout["adm3_dim_name"])


def _whole_day(value) -> int | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _check_rain_horizon(day_values, rain_day_max: int, where: str, *, allow_extra: bool) -> None:
    """A strict rain horizon needs the lead days 1..rain_day_max exactly (later
    days too, with allow_extra)."""
    parsed = [_whole_day(value) for value in day_values]
    invalid = [repr(value) for value, day in zip(day_values, parsed) if day is None]
    valid = [day for day in parsed if day is not None]
    duplicates = sorted({day for day in valid if valid.count(day) > 1})
    actual, required = set(valid), set(range(1, rain_day_max + 1))
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    problems = []
    if invalid:
        problems.append(f"non-integer day values={invalid[:10]}")
    if duplicates:
        problems.append(f"duplicate days={duplicates[:10]}")
    if missing:
        problems.append(f"missing days={missing[:10]}")
    if [day for day in extra if day < 1]:
        problems.append(f"nonpositive days={[day for day in extra if day < 1][:10]}")
    if [day for day in extra if day > rain_day_max] and not allow_extra:
        problems.append(f"extra days={[day for day in extra if day > rain_day_max][:10]}")
    if problems:
        requirement = (f"days 1..{rain_day_max}, with later days permitted" if allow_extra
                       else f"exactly days 1..{rain_day_max}")
        raise ValueError(f"{where}: strict rainfall horizon requires {requirement}; " + "; ".join(problems))


def _select_forecast_days(day_values, leads: LeadSelection, path) -> tuple[np.ndarray, list, list]:
    """(positions, labels, whole lead days) of the lead days the model reads
    (min_day..max_day), checking a strict rain horizon first."""
    day_values = np.asarray(day_values).reshape(-1)
    day_numbers = []
    for value in day_values:
        try:
            day_numbers.append(int(value))
        except (TypeError, ValueError):
            day_numbers.append(None)
    min_day, max_day, rain_day_max = leads.min_day, leads.max_day, leads.rain_day_max
    if rain_day_max is not None:
        policy = str(leads.rain_horizon_policy or "exact").strip().lower()
        _check_rain_horizon(
            [value for value, day in zip(day_values, day_numbers)
             if day is not None and (min_day is None or day >= int(min_day))],
            int(rain_day_max), f"NetCDF file {os.path.basename(path)}", allow_extra=policy == "truncate",
        )
    keep = np.ones(len(day_values), dtype=bool)
    if min_day is not None:
        keep &= np.array([day is not None and day >= int(min_day) for day in day_numbers])
    if max_day is not None:
        keep &= np.array([day is not None and day <= int(max_day) for day in day_numbers])
    if not np.any(keep):
        raise ValueError(f"Day filtering removed all columns for: {os.path.basename(path)}")
    positions = np.flatnonzero(keep)
    return positions, day_values[positions].tolist(), [day_numbers[position] for position in positions]


def _leading_tiles(v, n_times: int, n_members: int, member_dim, time_dim: int = 0):
    """(t0, t1, m0, m1) tiles over the time (and member) dimensions: whole
    storage chunks, grown along time to about _READ_SLAB_VALUES values, so
    each chunk is decompressed exactly once. member_dim and time_dim are the
    dimensions' positions in the variable (member_dim None without one)."""
    chunks = v.chunking()
    if isinstance(chunks, list):
        time_chunk = int(chunks[time_dim])
        member_step = int(chunks[member_dim]) if member_dim is not None else n_members
    else:  # contiguous storage: any slab reads only its own bytes
        time_chunk, member_step = 1, n_members
    member_step = max(1, min(n_members, member_step))
    per_time = int(np.prod(v.shape)) // max(1, v.shape[time_dim]) * member_step // max(1, n_members)
    rows = max(1, _READ_SLAB_VALUES // max(1, per_time))
    time_step = max(1, min(n_times, max(time_chunk, -(-rows // time_chunk) * time_chunk)))
    for t0 in range(0, n_times, time_step):
        for m0 in range(0, n_members, member_step):
            yield t0, min(n_times, t0 + time_step), m0, min(n_members, m0 + member_step)

