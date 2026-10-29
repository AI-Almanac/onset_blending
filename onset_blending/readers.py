"""Reading rainfall NetCDF files into arrays.

Values come out as the legacy table readers produce them (same dtype, NaN
for masked values, the same scaling), but as arrays over the file's
dimensions rather than one table row per value, and a forecast is read only
in the storage chunks that hold a kept point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._legacy import nc_utils


@dataclass
class ForecastArray:
    """A forecast file's rainfall at the kept points."""

    issues: np.ndarray            # datetime64[ns] issue dates (time of day dropped)
    members: np.ndarray | None    # member ids; None for a deterministic model
    points: pd.DataFrame          # the kept points, named as forecast_points() names them
    days: list[int]               # lead days the onset search reads
    rain: np.ndarray              # (issue, member, point, day); one member when deterministic
    member_spread: bool           # some spread-area point has two members for one issue and day


def _ranges(chunk: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(size, start + chunk)) for start in range(0, size, chunk)]


def read_forecast(path, variable: str, rename: dict, spec: dict, keep: np.ndarray, *,
                  spread_over: np.ndarray | None = None, scale: float = 1.0) -> ForecastArray | None:
    """The rainfall at the kept points (a mask over forecast_points()).

    Storage chunks are decompressed once each, and only where a kept point
    lies (or, until member spread is found, a spread-area point). None for a
    layout the point reader does not support."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if variable not in ds.variables:
            return None
        layout = nc_utils._forecast_layout(ds, variable, rename, "day")
        if layout is None:
            return None
        v = ds.variables[variable]
        dim_names, dim_vals = layout["dim_names"], layout["dim_vals"]
        day_idx, other_idx, n_leading = layout["day_idx"], layout["other_idx"], layout["n_leading"]
        day_positions, _, day_numbers = nc_utils._select_forecast_days(dim_vals[dim_names[day_idx]], spec, path)
        days = [int(d) for d in day_numbers]
        spatial = [dim_names[i] for i in other_idx[n_leading:]]
        shape = [len(dim_vals[d]) for d in spatial]
        n_points = int(np.prod(shape))
        keep = np.asarray(keep, dtype=bool)
        if keep.shape != (n_points,):
            raise ValueError(f"keep must have one entry per point ({n_points}).")
        kept = np.flatnonzero(keep)
        position = np.full(n_points, -1, dtype=np.int64)
        position[kept] = np.arange(len(kept))
        points = nc_utils._named_grid(
            nc_utils._product_frame(spatial, [dim_vals[d] for d in spatial]), rename, layout["adm3_dim_name"]
        ).take(kept).reset_index(drop=True)

        n_times = len(dim_vals[dim_names[0]])
        member_dim = other_idx[1] if layout["has_members"] else None
        n_members = len(dim_vals[dim_names[member_dim]]) if member_dim is not None else 1
        spread_area = np.zeros(n_points, dtype=bool)
        if spread_over is not None and member_dim is not None:
            spread_area[np.asarray(spread_over, dtype=bool)] = True
        member_spread = False
        present = None  # members present per (time, point, day), when chunks split the members

        chunks = v.chunking()
        spatial_axes = [dim_names.index(d) for d in spatial]
        tile_sizes = [int(chunks[a]) for a in spatial_axes] if isinstance(chunks, list) else shape
        grid_index = np.arange(n_points).reshape(shape)
        tiles = []
        for corner in np.ndindex(*[len(_ranges(t, s)) for t, s in zip(tile_sizes, shape)]):
            bounds = [_ranges(t, s)[c] for t, s, c in zip(tile_sizes, shape, corner)]
            tiles.append((bounds, grid_index[tuple(slice(a, b) for a, b in bounds)].reshape(-1)))

        rain = None
        perm = other_idx + [day_idx]
        for t0, t1, m0, m1 in nc_utils._leading_tiles(v, n_times, n_members, member_dim):
            for bounds, cells in tiles:
                kept_here = position[cells] >= 0
                area_here = spread_area[cells] if not member_spread else np.zeros(len(cells), dtype=bool)
                if not kept_here.any() and not area_here.any():
                    continue
                index = [slice(None)] * len(dim_names)
                index[0] = slice(t0, t1)
                if member_dim is not None:
                    index[member_dim] = slice(m0, m1)
                for axis, (a, b) in zip(spatial_axes, bounds):
                    index[axis] = slice(a, b)
                block = nc_utils._filled_with_nan(v[tuple(index)])
                block = np.transpose(block, perm)[..., day_positions]
                block = block.reshape(t1 - t0, m1 - m0, len(cells), len(days))
                if area_here.any():
                    counts = (~np.isnan(block[:, :, area_here, :])).sum(axis=1, dtype=np.int32)
                    if m1 - m0 < n_members:
                        if present is None:
                            present = np.zeros((n_times, n_points, len(days)), dtype=np.int32)
                        present[t0:t1, cells[area_here]] += counts
                        counts = present[t0:t1, cells[area_here]]
                    member_spread = bool((counts >= 2).any())
                if kept_here.any():
                    part = block[:, :, kept_here, :]
                    if scale != 1.0:
                        part = part * scale
                    if rain is None:
                        rain = np.empty((n_times, n_members, len(kept), len(days)), dtype=part.dtype)
                    rain[t0:t1, m0:m1, position[cells[kept_here]], :] = part
                del block
        if rain is None:
            rain = np.empty((n_times, n_members, 0, len(days)), dtype=np.float32)
        tinfo = nc_utils.get_nc_time(ds)
        stamps = pd.DatetimeIndex(nc_utils.nc_time_to_dates(dim_vals[dim_names[0]], tinfo["units"]))
        issues = stamps.normalize().to_numpy()
        members = np.asarray(dim_vals[dim_names[member_dim]]).astype(int) if member_dim is not None else None
    return ForecastArray(
        issues=issues, members=members, points=points, days=days, rain=rain, member_spread=member_spread
    )


@dataclass
class ObservationArray:
    """An observation file's rainfall, all points."""

    points: pd.DataFrame   # named as the legacy reader names them (lat/lon, or id)
    dates: np.ndarray      # datetime64[ns], one per time step
    years: np.ndarray      # the year of each time step
    rain: np.ndarray       # (time, point), the file's dtype, NaN where masked


def read_observations(path, variable: str, rename: dict) -> ObservationArray:
    """The rainfall of an observation file, as nc_read_groundtruth_long reads it
    but as a (time, point) array; the calendar must be daily without gaps."""
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
    dates = stamps.normalize().to_numpy()
    return ObservationArray(points=points, dates=dates, years=stamps.year.to_numpy(), rain=rain)
