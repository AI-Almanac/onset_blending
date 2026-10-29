"""Onsets: observed onset dates per unit and season, and forecast onset
probabilities per unit and issue date."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import _parallel, _specs, domain, formats, inputs, readers
from ._legacy import nc_utils, onset_utils
from .config import BlendConfig, ForecastModel
from .reference import OBSERVATIONS, Reference

# Peak bytes per rainfall value of one run of issues (the float32 run, the
# onset search's float64 copy of an issue, the summaries), and the
# interpreter's own footprint.
_FORECAST_BYTES_PER_RUN_VALUE = 10
_FORECAST_FIXED_BYTES = 512 * 2**20


def _context(config: BlendConfig, spec: dict, reference: Reference, source: str,
             area: domain.AreaFilter | None) -> dict:
    return {
        "spec": spec,
        "onset_params": onset_utils.read_onset_params(spec),
        "convention": domain.id_convention(config),
        "area": area if area is not None else domain.AreaFilter(config),
        "thresholds": reference.thresholds,
        "reference_onset": reference.reference_onset,
        "cell_transform": reference.cell_transforms.get(source),
        "regridder": reference.regridder(source),
    }


def _observed_file(path: Path, context: dict) -> pd.DataFrame:
    """One observation file's onsets: (id, year) rows with onset_day and onset_date.

    The file is read into one (unit, day) series per season and searched in
    one batched call; a cell transform, or two points sharing an id, take
    the legacy table route."""
    if context["cell_transform"] is not None:
        return _observed_file_from_table(path, context)
    read = readers.read_observations(
        path, context["spec"]["input"]["value_col"], context["spec"]["dimensions"]["rename"],
        shift_days=context["spec"]["options"].get("date_shift_days", 0),
    )
    rain_by_point, points = read.rain, read.points
    regridder = context["regridder"]
    if regridder is not None and "id" not in points.columns:
        transform = regridder.transform(*domain.grid_axes(points))
        source = domain.point_sources(points, transform)
        values = np.full((len(transform.source_ids), rain_by_point.shape[0]), np.nan)
        values[source[source >= 0]] = rain_by_point[:, source >= 0].T
        rain_by_point = domain.Regridder.apply(transform, values).T
        points = pd.DataFrame({"id": pd.Series(regridder.targets, dtype=str)})
    points = domain.with_ids(points, context["convention"])
    kept = np.zeros(len(points), dtype=bool)
    kept[context["area"](points.assign(_point=np.arange(len(points))))["_point"].to_numpy()] = True
    ids = points["id"].to_numpy()[kept]
    if pd.Index(ids).duplicated().any():
        return _observed_file_from_table(path, context)
    spec = context["spec"]
    policy = str(spec["options"]["missing_rain_policy"]).lower()
    rain = rain_by_point[:, kept].astype(float)
    if policy == "zero":
        rain = np.where(np.isnan(rain), 0.0, rain)

    order = np.argsort(ids.astype(str), kind="stable")
    ids = ids[order]
    rain = rain[:, order]
    thresholds = nc_utils.attach_thresholds_id(
        pd.DataFrame({"id": pd.Series(ids, dtype=str)}), context["thresholds"], spec=spec
    )["onset_thresh"].to_numpy(dtype=float)
    cutoff_md = spec["options"]["cutoff_month_day"]
    parts = []
    for year in np.unique(read.years):
        days = np.flatnonzero(read.years == year)
        days = days[np.argsort(read.dates[days], kind="stable")]
        series = rain[days].T
        units = np.ones(len(ids), dtype=bool)
        if policy == "drop":
            units = ~np.isnan(series).any(axis=1)
        cutoff = dt.date.fromisoformat(f"{year}-{cutoff_md}")
        ref = nc_utils.attach_ref_onset(
            pd.DataFrame({"id": pd.Series(ids[units], dtype=str), "year": np.full(units.sum(), year)}),
            context["reference_onset"],
        )["ref_onset_date"]
        start = np.array([max(cutoff, r) if pd.notna(r) else cutoff for r in ref], dtype=object)
        dates = read.dates[days]
        positions = np.searchsorted(dates.astype("datetime64[D]"), start.astype("datetime64[D]"), side="left")
        start_days = np.where(positions < len(dates), positions + 1, 9999)
        found = onset_utils.find_onsets_batch(
            series[units], thresholds[units], start_days, context["onset_params"], reject_if_short_followup=True
        )[0]
        has = np.isfinite(found)
        index = np.where(has, found, 1).astype(np.intp) - 1
        onset_dates = np.where(has, pd.DatetimeIndex(dates[index]).date, None)
        onset_day = np.where(has, (dates[index] - np.datetime64(cutoff, "D")).astype("timedelta64[D]").astype(float), np.nan)
        n = int(units.sum())
        parts.append(pd.DataFrame({
            "id": pd.Series(ids[units], dtype=str),
            "year": np.full(n, int(year), dtype=np.int64),
            "onset_idx": found,
            "onset_date": pd.Series(onset_dates, dtype=object),
            "onset_day": onset_day,
            "cutoff_date": pd.Series([cutoff] * n, dtype=object),
        }))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["id", "year"], kind="stable", ignore_index=True) if len(parts) > 1 else out


def _observed_file_from_table(path: Path, context: dict) -> pd.DataFrame:
    """The legacy route: a table row per unit and day, then the onset search."""
    spec = context["spec"]
    variable = spec["input"]["value_col"]
    df = nc_utils.nc_read_groundtruth_long(
        nc_path=str(path),
        var_name=variable,
        dim_rename_map=spec["dimensions"]["rename"],
        missing_rain_policy=spec["options"]["missing_rain_policy"],
    )
    if df is None:
        import netCDF4

        with netCDF4.Dataset(path) as ds:
            found = ", ".join(ds.variables)
        raise ValueError(f"{path.name} has no {variable!r} variable (it has: {found})")
    shift = int(spec["options"].get("date_shift_days", 0))
    if shift:
        # As read_observations does: dates onto the forecasts' labels, each day
        # keeping the season (year) of the calendar day it covers.
        codes, distinct = pd.factorize(df["time"])
        moved = (pd.to_datetime(pd.Index(distinct)) + pd.Timedelta(days=shift)).date
        df["time"] = pd.Series(np.asarray(moved, dtype=object)[codes], index=df.index, dtype=object)
    df = domain.with_ids(df, context["convention"])
    if context["cell_transform"] is not None:
        df = nc_utils.transform_groundtruth_long(df, context["cell_transform"], variable.lower(), spec=spec)
    df = context["area"](df)
    return nc_utils.process_ground_truth_rainfall_id(
        df,
        spec,
        ref_onset_dt=context["reference_onset"],
        thr_dt=context["thresholds"],
        value_col=variable.lower(),
    )["wide"]


@dataclass
class ObservedOnsets:
    """Observed onsets over (unit, year): onset_day (days after the season
    start; NaN when there is none), onset_idx (the day in its season's file,
    1-based), and present (the unit had a series that year)."""

    data: xr.Dataset
    season_start: str

    @classmethod
    def from_frames(cls, parts: list[pd.DataFrame], season_start: str) -> "ObservedOnsets":
        """From per-file tables (id, year, onset_idx, onset_day)."""
        units = np.unique(np.concatenate([p["id"].to_numpy(dtype=str) for p in parts]))
        years = np.unique(np.concatenate([p["year"].to_numpy(dtype=np.int64) for p in parts]))
        shape = (len(units), len(years))
        onset_day, onset_idx = np.full(shape, np.nan), np.full(shape, np.nan)
        present = np.zeros(shape, dtype=bool)
        for part in parts:
            u = np.searchsorted(units, part["id"].to_numpy(dtype=str))
            y = np.searchsorted(years, part["year"].to_numpy(dtype=np.int64))
            if present[u, y].any():
                raise ValueError("Two observation files hold the same unit and season")
            onset_day[u, y] = part["onset_day"].to_numpy(dtype=float)
            onset_idx[u, y] = part["onset_idx"].to_numpy(dtype=float)
            present[u, y] = True
        data = xr.Dataset(
            {"onset_day": (("unit", "year"), onset_day), "onset_idx": (("unit", "year"), onset_idx),
             "present": (("unit", "year"), present)},
            coords={"unit": units.astype(object), "year": years},
        )
        return cls(data=data, season_start=season_start)

    @property
    def years(self) -> list[int]:
        return [int(y) for y in self.data["year"].values]

    def cutoff_dates(self) -> np.ndarray:
        """The season start of each year, datetime64[D]."""
        return np.array([np.datetime64(f"{y}-{self.season_start}") for y in self.years], dtype="datetime64[D]")

    def onset_dates(self) -> np.ndarray:
        """(unit, year) onset dates, datetime64[D]; NaT where there is none."""
        day = self.data["onset_day"].values
        dates = self.cutoff_dates()[np.newaxis, :] + np.where(np.isnan(day), 0, day).astype("timedelta64[D]")
        return np.where(np.isnan(day), np.datetime64("NaT"), dates)

    def onsets(self) -> pd.DataFrame:
        """(id, year, onset_day) for each present unit and season, season by
        season, units in order: what the climatology is fitted on."""
        present = self.data["present"].values.T
        y, u = np.nonzero(present)
        return pd.DataFrame({
            "id": pd.Series(self.data["unit"].values[u], dtype=str),
            "year": self.data["year"].values[y].astype(np.int64),
            "onset_day": self.data["onset_day"].values.T[y, u],
        })

    def frame(self) -> pd.DataFrame:
        """The legacy table: one row per present unit and season."""
        onsets = self.onsets()
        present = self.data["present"].values.T
        y, u = np.nonzero(present)
        dates = self.onset_dates().T[y, u]
        cutoffs = self.cutoff_dates()[y]
        return pd.DataFrame({
            "id": onsets["id"],
            "year": onsets["year"],
            "onset_idx": self.data["onset_idx"].values.T[y, u],
            "onset_date": pd.Series(_as_dates(dates.astype("datetime64[ns]")), dtype=object).where(~np.isnat(dates), None),
            "onset_day": onsets["onset_day"],
            "cutoff_date": pd.Series(_as_dates(cutoffs.astype("datetime64[ns]")), dtype=object),
        })

    def blendable(self, min_onset_years: int) -> frozenset[str]:
        """Units with at least min_onset_years observed onsets over every season.

        The climatology keeps only units with that many onsets in its training
        years, and the inner join drops every forecast unit the climatology
        lacks, so no forecast unit outside this set reaches the features."""
        counts = (~np.isnan(self.data["onset_day"].values)).sum(axis=1)
        return frozenset(self.data["unit"].values[counts >= int(min_onset_years)].astype(str))


def observed(config: BlendConfig, *, reference: Reference | None = None, workers: int = 1,
             budget_bytes: int | None = None) -> ObservedOnsets:
    """Observed onsets for every observation file."""
    paths = inputs.list_files(config.observations.files, config, "observations")
    reference = reference or Reference.resolve(config)
    context = _context(config, _specs.observation_spec(config), reference, OBSERVATIONS, None)
    parts = _parallel.map_ordered(_observed_file, paths, context, workers)
    return ObservedOnsets.from_frames(parts, config.onset.season_start)


# Onset-search variants: the search start (lead 1, the fixed cutoff date, the
# reference onset) and the legacy column each probability series fills.
VARIANTS = ("base", "fixed_cutoff", "ref")
_LEGACY_PREFIX = {"base": "predicted_prob", "fixed_cutoff": "predicted_prob_fixed_cutoff",
                  "ref": "predicted_prob_ref"}


@dataclass
class ForecastOnsets:
    """A model's forecast onsets as arrays over (issue, unit, day):

    - count_<variant>: members whose onset falls on each lead day (uint8);
    - members: members per issue;
    - rain_mean: the members' mean rainfall;
    - onset_threshold (unit);
    - for the day-level table only: rain_sd, raining (members with more than
      1 mm, uint8) and ref_onset_date (issue, unit).

    Probabilities and fractions are counts over members, computed on demand.
    """

    model: str
    data: xr.Dataset | None
    has_spread: bool = False
    years: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.years and self.data is not None:
            self.years = sorted({int(y) for y in self.data["year"].values})

    def probability(self, variant: str = "base") -> xr.DataArray:
        """The share of members with an onset on each lead day."""
        return self.data[f"count_{variant}"] / self.data["members"].astype(float)

    def frac_raining(self) -> xr.DataArray:
        return self.data["raining"] / self.data["members"].astype(float)

    def release(self) -> None:
        """Drop the arrays, keeping the summary."""
        self.data = None

    def legacy_parts(self) -> list[pd.DataFrame]:
        """The per-season tables the legacy pipeline produced, one per year."""
        return [_legacy_part(self, year) for year in self.years]


def _legacy_part(onsets: ForecastOnsets, year: int) -> pd.DataFrame:
    data = onsets.data.isel(issue=np.flatnonzero(onsets.data["year"].values == year))
    units = data["unit"].values.astype(str)
    order = np.argsort(units, kind="stable")
    issues = pd.DatetimeIndex(data["issue"].values)
    n_issue, n_unit = len(issues), len(units)
    unit_rows = np.repeat(order, n_issue)
    issue_rows = np.tile(np.arange(n_issue), n_unit)
    days = data["day"].values
    columns = {
        "id": pd.Series(units[unit_rows], dtype=str),
        "time": pd.Series(issues.date[issue_rows], dtype=object),
        "year": pd.Series(np.full(len(unit_rows), year, dtype=np.int64)),
        "onset_thresh": pd.Series(data["onset_threshold"].values[unit_rows]),
        "ref_onset_date": pd.Series(_as_dates(data["ref_onset_date"].values[issue_rows, unit_rows]), dtype=object),
    }
    values = {
        "forecast_rain": data["rain_mean"].values,
        "forecast_rain_sd": data["rain_sd"].values if "rain_sd" in data else None,
        "frac_raining": (data["raining"] / data["members"].astype(float)).values,
        **{_LEGACY_PREFIX[v]: (data[f"count_{v}"] / data["members"].astype(float)).values for v in VARIANTS},
    }
    frames = [pd.DataFrame(columns)]
    for prefix, array in values.items():
        if array is None:
            array = np.full((n_issue, n_unit, len(days)), np.nan)
        frames.append(pd.DataFrame(
            array[issue_rows, unit_rows], columns=[f"{prefix}_day_{d}" for d in days], copy=False
        ))
    return pd.concat(frames, axis=1)


def _as_dates(values: np.ndarray) -> np.ndarray:
    """datetime64 values as date objects, NaT as NaT, each distinct value once."""
    codes, distinct = pd.factorize(values, use_na_sentinel=False)
    stamps = pd.DatetimeIndex(distinct)
    out = np.array([pd.NaT if pd.isna(t) else t.date() for t in stamps], dtype=object)
    return out[codes]


def _kept_mask(path: Path, context: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """The file's points with ids, the points kept (in the area, and among
    keep_ids), and the points of the area (where member spread is judged)."""
    points = nc_utils.forecast_points(str(path), context["variable"], context["spec"]["dimensions"]["rename"], day_dim="day")
    if points is None:
        return None
    points = domain.with_ids(points, context["convention"])
    inside = context["area"].area(points.assign(_point=np.arange(len(points))))
    in_area = np.zeros(len(points), dtype=bool)
    in_area[inside["_point"].to_numpy()] = True
    if points["id"][in_area].duplicated().any():
        return None
    keep = in_area.copy()
    if context["keep_ids"] is not None:
        keep &= points["id"].isin(context["keep_ids"]).to_numpy()
    return points, keep, in_area


def _unit_frame_values(context: dict, ids: np.ndarray, issues: np.ndarray):
    """Per-unit thresholds and per-(issue, unit) reference onset dates, as the
    legacy pipeline attaches them to its rows."""
    spec = context["spec"]
    thresholds = nc_utils.attach_thresholds_id(
        pd.DataFrame({"id": pd.Series(ids, dtype=str)}), context["thresholds"], spec=spec
    )["onset_thresh"].to_numpy(dtype=float)
    issue_index = pd.DatetimeIndex(issues)
    pairs = pd.DataFrame({
        "id": pd.Series(np.tile(ids, len(issues)), dtype=str),
        "time": np.repeat(issue_index.date, len(ids)),
        "year": np.repeat(issue_index.year.to_numpy(dtype=int), len(ids)),
    })
    with_ref = nc_utils.attach_ref_onset(pairs, context["reference_onset"])
    ref = pd.to_datetime(with_ref["ref_onset_date"]).to_numpy().reshape(len(issues), len(ids))
    return thresholds, ref


def _file_context(path: Path, context: dict) -> dict:
    """The context for one file, read in its own format: its dimension names
    and the scale that puts its rainfall in millimetres (see formats)."""
    fmt = formats.of_file(path, context["variable"], context["dims"], context["fallback_unit_scale"])
    dimensions = context["spec"]["dimensions"]
    spec = {**context["spec"], "dimensions": {**dimensions, "rename": {**dimensions["rename"], **fmt.rename}}}
    return {**context, "spec": spec, "unit_scale": fmt.scale}


def _forecast_file(path: Path, context: dict) -> tuple[xr.Dataset, bool]:
    """One forecast file's onset counts and rain summaries, a run of issues
    at a time."""
    context = _file_context(path, context)
    if context["cell_transform"] is not None or context["area"].dissemination:
        return _forecast_file_from_table(path, context)
    if context["regridder"] is not None:
        points = nc_utils.forecast_points(str(path), context["variable"], context["spec"]["dimensions"]["rename"], day_dim="day")
        if points is not None and "id" not in points.columns:
            return _regridded_forecast_file(path, points, context)
    masks = _kept_mask(path, context)
    if masks is None:
        return _forecast_file_from_table(path, context)
    points, keep, in_area = masks
    spec = context["spec"]
    with readers.ForecastFile(
        path, context["variable"], spec["dimensions"]["rename"], spec, keep,
        spread_over=in_area, scale=context["unit_scale"],
    ) as forecast:
        selected = _selected_members(forecast.members, context, path)
        summary = _Summary(points["id"].to_numpy()[keep], forecast.issues, forecast.days, context)
        for start, stop, rain in forecast.issue_runs():
            summary.add(start, stop, rain[:, selected])
        return summary.dataset(), forecast.member_spread


def _selected_members(members: np.ndarray | None, context: dict, path: Path):
    """The members read: up to max_member, and only member_ids when given
    (a forecast reads the members the blend was trained on)."""
    if members is None or (context["max_member"] is None and context["member_ids"] is None):
        return slice(None)
    selected = np.ones(len(members), dtype=bool)
    if context["max_member"] is not None:
        selected &= members <= context["max_member"]
    if context["member_ids"] is not None:
        selected &= np.isin(members, context["member_ids"])
    if not selected.any():
        raise ValueError(f"{path.name}: none of its members are read")
    return selected


def _regridded_forecast_file(path: Path, points: pd.DataFrame, context: dict) -> tuple[xr.Dataset, bool]:
    """A gridded forecast summarised on the blend's units: each run of issues
    is read at the cells feeding the kept units and regridded in memory, then
    scaled (the legacy order: regrid the file, scale as it is read)."""
    regridder = context["regridder"]
    transform = regridder.transform(*domain.grid_axes(points))
    source = domain.point_sources(points, transform)
    units = pd.DataFrame({"id": pd.Series(regridder.targets, dtype=str)})
    in_area = np.zeros(len(units), dtype=bool)
    in_area[context["area"].area(units.assign(_point=np.arange(len(units))))["_point"].to_numpy()] = True
    keep = in_area & (units["id"].isin(context["keep_ids"]).to_numpy() if context["keep_ids"] is not None else True)
    kept_units = np.flatnonzero(keep)
    feeding = np.isin(source, domain.Regridder.sources_of(transform, kept_units))
    spec = context["spec"]
    with readers.ForecastFile(path, context["variable"], spec["dimensions"]["rename"], spec, feeding) as forecast:
        selected = _selected_members(forecast.members, context, path)
        rows = source[feeding]
        summary = _Summary(units["id"].to_numpy()[keep], forecast.issues, forecast.days, context)
        member_spread = False
        for start, stop, rain in forecast.issue_runs():
            n_run, n_member, _, n_day = rain.shape
            values = np.full((len(transform.source_ids), n_run * n_member * n_day), np.nan)
            values[rows] = rain.transpose(2, 0, 1, 3).reshape(len(rows), -1)
            regridded = domain.Regridder.apply(transform, values)[kept_units]
            regridded = regridded.reshape(len(kept_units), n_run, n_member, n_day).transpose(1, 2, 0, 3)
            if context["unit_scale"] != 1.0:
                regridded = regridded * context["unit_scale"]
            if forecast.members is not None and not member_spread:
                member_spread = bool(((~np.isnan(regridded)).sum(axis=1) >= 2).any())
            summary.add(start, stop, np.ascontiguousarray(regridded[:, selected]))
    return summary.dataset(), member_spread


def _summarise(rain: np.ndarray, ids: np.ndarray, issues: np.ndarray, days: list[int], context: dict) -> xr.Dataset:
    """Onset counts and rain summaries from a file's (issue, member, unit, day) rainfall."""
    summary = _Summary(ids, issues, days, context)
    summary.add(0, len(issues), rain)
    return summary.dataset()


def member_mean(rain: np.ndarray) -> np.ndarray:
    """(issue, member, unit, day) rainfall's mean over members, as the legacy
    pipeline computed it with pandas' compiled grouped mean: a compensated
    (Kahan) sum over the members in order, in the rainfall's own dtype,
    skipping NaN. (pandas' float64 fast path, taken only for sorted groups
    without NaN, never applied to the legacy rows.)"""
    shape = (rain.shape[0],) + rain.shape[2:]
    total = np.zeros(shape, rain.dtype)
    compensation = np.zeros(shape, rain.dtype)
    count = np.zeros(shape, np.int64)
    for member in range(rain.shape[1]):
        value = rain[:, member]
        present = ~np.isnan(value)
        y = value - compensation
        t = total + y
        c = (t - total) - y
        c[np.isnan(c)] = 0  # an infinite value
        np.copyto(compensation, c, where=present)
        np.copyto(total, t, where=present)
        count += present
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / count.astype(rain.dtype)
    out[count == 0] = np.nan
    return out


class _Summary:
    """A file's onset counts and rain summaries, filled a run of issues at a time."""

    def __init__(self, ids: np.ndarray, issues: np.ndarray, days: list[int], context: dict):
        self.ids, self.issues, self.days, self.context = ids, issues, days, context
        n_issue, n_unit, n_day = len(issues), len(ids), len(days)
        self.thresholds, self.ref = _unit_frame_values(context, ids, issues)
        issue_rows = np.repeat(issues, n_unit)
        fixed, from_ref = nc_utils._forecast_start_days(
            issue_rows, pd.DatetimeIndex(issue_rows).year, self.ref.reshape(-1), np.asarray(days),
            context["spec"]["options"]["fixed_cutoff_month_day"],
        )
        self.fixed, self.from_ref = fixed.reshape(n_issue, n_unit), from_ref.reshape(n_issue, n_unit)
        self.counts = {v: np.zeros((n_issue, n_unit, n_day), dtype=np.uint8) for v in VARIANTS}
        self.members = np.zeros(n_issue, dtype=np.uint8)
        self.rain_mean = self.rain_sd = self.raining = None

    def add(self, start: int, stop: int, rain: np.ndarray) -> None:
        params = self.context["onset_params"]
        n_run, n_member, n_unit, n_day = rain.shape
        units = np.tile(np.arange(n_unit), n_member)
        thresholds = np.tile(self.thresholds, n_member)
        for offset in range(n_run):
            t = start + offset
            starts = np.vstack((
                np.ones(n_member * n_unit, dtype=int), np.tile(self.fixed[t], n_member),
                np.tile(self.from_ref[t], n_member),
            ))
            found = onset_utils.find_onsets_batch(
                rain[offset].reshape(n_member * n_unit, n_day), thresholds, starts, params
            )
            for variant, onset in zip(VARIANTS, found):
                valid = np.isfinite(onset) & (onset >= 1) & (onset <= n_day)
                np.add.at(self.counts[variant][t], (units[valid], onset[valid].astype(np.intp) - 1), 1)
        self.members[start:stop] = n_member

        if self.rain_mean is None:
            self.rain_mean = np.empty((len(self.issues), n_unit, n_day), dtype=rain.dtype)
        # One member is its own mean, with no spread.
        self.rain_mean[start:stop] = rain[:, 0] if n_member == 1 else member_mean(rain)
        if self.context["daily"]:
            if self.raining is None:
                self.raining = np.zeros((len(self.issues), n_unit, n_day), dtype=np.uint8)
                self.rain_sd = np.full((len(self.issues), n_unit, n_day), np.nan, dtype=rain.dtype)
            self.raining[start:stop] = (rain > 1).sum(axis=1)
            if n_member > 1:
                # pandas' grouped std (its compiled kernel, in member order).
                rows = rain.reshape(n_run * n_member * n_unit, n_day)
                groups = (np.repeat(np.arange(n_run), n_member * n_unit) * n_unit
                          + np.tile(np.arange(n_unit), n_run * n_member))
                std = pd.DataFrame(rows, copy=False).groupby(groups, sort=True).std()
                self.rain_sd[start:stop] = std.to_numpy().reshape(n_run, n_unit, n_day)

    def dataset(self) -> xr.Dataset:
        variables = {
            "rain_mean": (("issue", "unit", "day"), self.rain_mean),
            "members": (("issue",), self.members),
            "onset_threshold": (("unit",), self.thresholds),
            "year": (("issue",), pd.DatetimeIndex(self.issues).year.to_numpy(dtype=np.int64)),
            **{f"count_{v}": (("issue", "unit", "day"), self.counts[v]) for v in VARIANTS},
        }
        if self.context["daily"]:
            # What only the day-level table reports.
            variables["rain_sd"] = (("issue", "unit", "day"), self.rain_sd)
            variables["raining"] = (("issue", "unit", "day"), self.raining)
            variables["ref_onset_date"] = (("issue", "unit"), self.ref)
        return xr.Dataset(
            variables, coords={"issue": self.issues, "unit": self.ids.astype(object), "day": np.asarray(self.days)}
        )


def _forecast_file_from_table(path: Path, context: dict) -> tuple[xr.Dataset, bool]:
    """The legacy table route, for cell transforms and dissemination filters:
    read every row, transform and filter, then summarise as arrays."""
    spec = context["spec"]
    df = nc_utils.nc_read_forecast_wide(
        nc_path=str(path), var_name=context["variable"], dim_rename_map=spec["dimensions"]["rename"],
        spec=spec, day_dim="day", prefix="rain",
    )
    if df is None:
        raise ValueError(f"{path.name} has no {context['variable']!r} variable")
    rain_cols = [c for c in df.columns if c.startswith("rain_day_")]
    _, ordered = readers.lead_days_in_order(
        np.arange(len(rain_cols)), [int(c.rsplit("_", 1)[1]) for c in rain_cols], path.name
    )
    rain_cols = [f"rain_day_{d}" for d in ordered]
    if context["unit_scale"] != 1.0:
        df[rain_cols] = df[rain_cols].to_numpy() * context["unit_scale"]
    df = domain.with_ids(df, context["convention"])
    if context["cell_transform"] is not None:
        df = nc_utils.transform_forecast_wide(df, context["cell_transform"], spec=spec)
    df = context["area"](df)
    has_spread = "number" in df.columns and bool(
        (df[rain_cols].notna().groupby([df["id"], df["time"]]).sum() >= 2).to_numpy().any()
    )
    if context["keep_ids"] is not None:
        df = df[df["id"].isin(context["keep_ids"])]
    if "number" in df.columns and context["max_member"] is not None:
        df = df[df["number"] <= context["max_member"]]
    if "number" in df.columns and context["member_ids"] is not None:
        df = df[df["number"].isin(context["member_ids"])]
    issues = np.sort(pd.to_datetime(df["time"].unique()).to_numpy())
    ids = np.sort(df["id"].astype(str).unique())
    members = np.sort(df["number"].unique()) if "number" in df.columns else np.array([0])
    days = [int(c.rsplit("_", 1)[1]) for c in rain_cols]
    rain = np.full((len(issues), len(members), len(ids), len(days)), np.nan, dtype=df[rain_cols[0]].dtype)
    t = np.searchsorted(issues, pd.to_datetime(df["time"]).to_numpy())
    m = np.searchsorted(members, df["number"].to_numpy()) if "number" in df.columns else np.zeros(len(df), int)
    u = np.searchsorted(ids, df["id"].astype(str).to_numpy())
    rain[t, m, u] = df[rain_cols].to_numpy()
    return _summarise(rain, ids, issues, days, context), has_spread


def _forecast_bytes(path: Path, context: dict) -> int:
    """Peak memory to process one forecast file: one run of issues' rainfall
    at the kept points (read in storage-chunk runs) and its working copies,
    over the interpreter's own. Measured at 1.04 GB for an India ENS file
    (26 members, 4,624 kept units) against this estimate's 1.1 GB."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        variable = ds.variables.get(context["variable"])
        if variable is None:
            return _FORECAST_FIXED_BYTES
    context = _file_context(path, context)
    with netCDF4.Dataset(path) as ds:
        variable = ds.variables[context["variable"]]
        values = math.prod(variable.shape)
        chunks = variable.chunking()
        run = int(chunks[0]) if isinstance(chunks, list) else 1
        n_times = variable.shape[0]
    kept = 1.0
    if context["keep_ids"] is not None:
        points = nc_utils.forecast_points(
            str(path), context["variable"], context["spec"]["dimensions"]["rename"], day_dim="day"
        )
        if points is not None and len(points):
            kept = float(domain.with_ids(points, context["convention"])["id"].isin(context["keep_ids"]).mean())
    run_values = values * kept * min(1.0, run / max(n_times, 1))
    return int(run_values * _FORECAST_BYTES_PER_RUN_VALUE) + _FORECAST_FIXED_BYTES


def forecast(
    config: BlendConfig,
    model: ForecastModel,
    *,
    keep_ids: frozenset[str] | None = None,
    daily: bool = False,
    reference: Reference | None = None,
    area: domain.AreaFilter | None = None,
    member_ids: list[int] | None = None,
    workers: int = 1,
    budget_bytes: int | None = None,
) -> ForecastOnsets:
    """A model's forecast onsets for every one of its files.

    keep_ids, when given, limits the units to those (the blendable units
    under the inner join); the results for them are unchanged. daily also
    keeps what only the day-level table reports: the members' rain spread,
    the raining fraction and the reference onset dates. reference and area
    default to the config's (a forecast passes the trained blend's);
    member_ids limits the members read (a forecast reads the trained ones).
    """
    paths = inputs.list_files(model.files, config, f"model {model.name}")
    spec = _specs.forecast_spec(config, model)
    context = {
        **_context(config, spec, reference or Reference.resolve(config), model.name, area),
        "variable": model.variable,
        # Each file's own dimension names and units decide how it is read;
        # these apply where a file does not say (see _file_context).
        "dims": dict(model.dims),
        "fallback_unit_scale": float(model.unit_scale),
        "keep_ids": keep_ids,
        "daily": daily,
        "max_member": model.max_member,
        "member_ids": None if member_ids is None else [int(m) for m in member_ids],
    }
    count = _parallel.workers_within([_forecast_bytes(p, context) for p in paths], workers, budget_bytes)
    results = _parallel.map_ordered(_forecast_file, paths, context, count)
    units = [tuple(data["unit"].values) for data, _ in results]
    if len(set(units)) > 1:
        raise ValueError(f"model {model.name}: its files cover different units")
    # Per-unit variables are the same in every file; only issue variables stack.
    data = (
        xr.concat([data for data, _ in results], dim="issue", data_vars="minimal", coords="minimal",
                  compat="equals", join="exact")
        if results
        else None
    )
    if data is not None:
        # Issues are looked up by date later, so they must be distinct and in
        # order, whatever order the files hold them in. Issue times of day
        # fold onto their date (the signature records the hour).
        issues = pd.DatetimeIndex(data["issue"].values)
        repeated = sorted({d.date().isoformat() for d in issues[issues.duplicated()]})
        if repeated:
            raise ValueError(
                f"model {model.name}: several issues on the same date ({', '.join(repeated[:5])}"
                f"{'…' if len(repeated) > 5 else ''}): one initialisation per date"
            )
        if not issues.is_monotonic_increasing:
            data = data.sortby("issue")
    return ForecastOnsets(model=model.name, data=data, has_spread=any(spread for _, spread in results))
