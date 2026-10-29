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

from . import _specs, domain, ids, io, parallel
from ..config import BlendConfig, ForecastModel
from .onset_rule import OnsetRule, find_onsets
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
        "onset_rule": OnsetRule.of(config.onset.definition),
        "convention": domain.id_convention(config),
        "area": area if area is not None else domain.AreaFilter(config),
        "reference": reference,
        "thresholds": reference.thresholds,
        "reference_onset": reference.reference_onset,
        "cell_transform": reference.cell_transforms.get(source),
        "regridder": reference.regridder(source),
    }


def _observed_file(path: Path, context: dict) -> pd.DataFrame:
    """One observation file's onsets: (id, year) rows with onset_day and onset_date.

    The file is read into one (unit, day) series per season and searched in
    one batched call. Under a cell transform the missing-rain policy applies
    to its source series (a dropped source leaves its targets for that
    season, or the target itself when it has no other), as the legacy
    pipeline applied it on reading."""
    spec = context["spec"]
    policy = str(spec["options"]["missing_rain_policy"]).lower()
    read = io.read_observations(
        path, spec["input"]["value_col"], spec["dimensions"]["rename"],
        shift_days=spec["options"].get("date_shift_days", 0),
    )
    rain_by_point, points = read.rain, read.points
    years = np.unique(read.years)
    supported = None  # (unit, season) a cell transform's targets have sources for
    regridder = context["regridder"]
    if context["cell_transform"] is not None:
        transform, source = _cell_transform(points, context["cell_transform"], context["convention"], path.name)
        values = np.full((len(transform.source_ids), rain_by_point.shape[0]), np.nan)
        values[source[source >= 0]] = rain_by_point[:, source >= 0].T.astype(float)
        present = np.zeros((len(transform.source_ids), len(years)), dtype=bool)
        present[source[source >= 0]] = True
        if policy == "zero":
            values = np.where(np.isnan(values), 0.0, values)
        elif policy == "drop":
            for y, year in enumerate(years):
                days = read.years == year
                incomplete = np.isnan(values[:, days]).any(axis=1)
                values[np.ix_(incomplete, days)] = np.nan
                present[incomplete, y] = False
        rain_by_point = transform.observed_mean(values).T
        supported = transform.supported(present)
        points = pd.DataFrame({"id": pd.Series(transform.target_ids, dtype=str)})
        policy = "keep"  # applied to the sources
    elif regridder is not None and "id" not in points.columns:
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
    _refuse_shared_ids(ids, path)
    rain = rain_by_point[:, kept].astype(float)
    if policy == "zero":
        rain = np.where(np.isnan(rain), 0.0, rain)

    order = np.argsort(ids.astype(str), kind="stable")
    ids = ids[order]
    rain = rain[:, order]
    if supported is not None:
        supported = supported[kept][order]
    thresholds = context["reference"].unit_thresholds(ids.astype(str), context["convention"])
    cutoff_md = spec["options"]["cutoff_month_day"]
    parts = []
    for y, year in enumerate(years):
        days = np.flatnonzero(read.years == year)
        days = days[np.argsort(read.dates[days], kind="stable")]
        series = rain[days].T
        units = np.ones(len(ids), dtype=bool) if supported is None else supported[:, y].copy()
        if policy == "drop":
            units = ~np.isnan(series).any(axis=1)
        cutoff = dt.date.fromisoformat(f"{year}-{cutoff_md}")
        ref = context["reference"].reference_dates(ids[units].astype(str), np.full(units.sum(), year))
        start = np.array([max(cutoff, pd.Timestamp(r).date()) if not np.isnat(r) else cutoff for r in ref], dtype=object)
        dates = read.dates[days]
        positions = np.searchsorted(dates.astype("datetime64[D]"), start.astype("datetime64[D]"), side="left")
        start_days = np.where(positions < len(dates), positions + 1, 9999)
        found = find_onsets(
            series[units], thresholds[units], start_days, context["onset_rule"], reject_if_short_followup=True
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


def _refuse_shared_ids(ids: np.ndarray, path: Path) -> None:
    """Two points of a file under one unit id have no single series."""
    shared = pd.Index(ids.astype(str))
    shared = shared[shared.duplicated()].unique()
    if len(shared):
        raise ValueError(f"{path.name}: points share the unit ids {', '.join(map(str, shared[:5]))}")


def _cell_transform(points: pd.DataFrame, weights: pd.DataFrame, explicit, where: str):
    """The cell transform from a file's points onto its targets, and each
    point's source position (-1 for a point the weights do not use). Points
    are named as the weights' source ids are: grid ids under the convention
    those establish, else their own ids."""
    expected = ids.normalize(weights["source_id"].drop_duplicates(), context="cell-transform source IDs")
    convention = ids.resolve(explicit, authoritative_ids=expected, context=f"cell transform for {where}")
    if convention is not None and {"lat", "lon"} <= set(points.columns):
        point_ids = np.asarray(ids.format_grid_ids(
            pd.to_numeric(points["lat"]).to_numpy(float), pd.to_numeric(points["lon"]).to_numpy(float), convention
        ), dtype=object)
    else:
        point_ids = ids.with_id_column(points, explicit=explicit, context="cell transform")["id"].to_numpy(dtype=object)
    missing = sorted(set(expected) - set(point_ids.tolist()))
    if missing:
        raise ValueError(
            f"{where}: {len(missing)} expected weight source IDs are absent from the raw NetCDF coordinates: "
            f"{', '.join(missing[:10])}"
        )
    transform = domain.SparseTransform.compile(
        weights, source_ids=tuple(sorted(weights["source_id"].unique())),
        target_ids=tuple(sorted(weights["target_id"].unique())),
    )
    source = np.array([transform.source_index.get(i, -1) for i in point_ids.tolist()], dtype=np.int64)
    used = source[source >= 0]
    if len(np.unique(used)) != len(used):
        raise ValueError(f"{where}: points share cell-transform source ids")
    return transform, source


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
    paths = io.list_files(config.observations.files, config, "observations")
    reference = reference or Reference.resolve(config)
    context = _context(config, _specs.observation_spec(config), reference, OBSERVATIONS, None)
    parts = parallel.map_ordered(_observed_file, paths, context, workers)
    return ObservedOnsets.from_frames(parts, config.onset.season_start)


# Onset-search variants: the search starts at lead 1, the fixed cutoff date,
# or the reference onset.
VARIANTS = ("base", "fixed_cutoff", "ref")


@dataclass
class ForecastOnsets:
    """A model's forecast onsets as arrays over (issue, unit, day):

    - count_<variant>: members whose onset falls on each lead day (uint8);
    - members: members per issue;
    - rain_mean: the members' mean rainfall;
    - onset_threshold (unit).

    Probabilities are counts over members, computed on demand; has_spread
    says whether any unit's members differ.
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

    def release(self) -> None:
        """Drop the arrays, keeping the summary."""
        self.data = None


def _as_dates(values: np.ndarray) -> np.ndarray:
    """datetime64 values as date objects, NaT as NaT, each distinct value once."""
    codes, distinct = pd.factorize(values, use_na_sentinel=False)
    stamps = pd.DatetimeIndex(distinct)
    out = np.array([pd.NaT if pd.isna(t) else t.date() for t in stamps], dtype=object)
    return out[codes]


def _kept_mask(points: pd.DataFrame, context: dict, path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The file's points with ids, the points kept (in the area and the
    dissemination units, and among keep_ids), and the points of the area
    (where member spread is judged)."""
    points = domain.with_ids(points, context["convention"])
    in_area = _inside(context["area"], points["id"])
    _refuse_shared_ids(points["id"].to_numpy()[in_area], path)
    keep = in_area.copy()
    if context["keep_ids"] is not None:
        keep &= points["id"].isin(context["keep_ids"]).to_numpy()
    return points, keep, in_area


def _inside(area: domain.AreaFilter, unit_ids) -> np.ndarray:
    """Per unit: inside the area of interest (and the dissemination units
    when they restrict the inputs)."""
    units = pd.DataFrame({"id": pd.Series(unit_ids, dtype=str)})
    inside = np.zeros(len(units), dtype=bool)
    inside[area(units.assign(_point=np.arange(len(units))))["_point"].to_numpy()] = True
    return inside


def _unit_frame_values(context: dict, ids: np.ndarray, issues: np.ndarray):
    """Per-unit thresholds and per-(issue, unit) reference onset dates."""
    reference = context["reference"]
    thresholds = reference.unit_thresholds(ids.astype(str), context["convention"])
    years = pd.DatetimeIndex(issues).year.to_numpy(dtype=int)
    ref = reference.reference_dates(np.tile(ids.astype(str), len(issues)), np.repeat(years, len(ids)))
    return thresholds, ref.reshape(len(issues), len(ids))


def _start_days(issue_dates: np.ndarray, ref_dates: np.ndarray, days: np.ndarray, fixed_cutoff: str):
    """The first lead day an onset may fall on, per issue: on or after the
    fixed cutoff date, and on or after the reference onset (1 without one);
    9999 past the last lead day."""
    issue_dates = pd.to_datetime(issue_dates).values
    days = np.asarray(days)
    years = pd.DatetimeIndex(issue_dates).year.to_numpy(dtype=int)
    distinct, codes = np.unique(years, return_inverse=True)
    cutoffs = np.array([np.datetime64(f"{year}-{fixed_cutoff}") for year in distinct])[codes.reshape(-1)]
    offsets = (cutoffs - issue_dates).astype("timedelta64[D]").astype(int)
    fixed = np.searchsorted(days, offsets - 1, side="right") + 1
    fixed[fixed > len(days)] = 9999
    ref_dates = pd.to_datetime(ref_dates).values
    from_ref = np.ones(len(issue_dates), dtype=int)
    has_ref = ~pd.isnull(ref_dates)
    if np.any(has_ref):
        ref_offsets = (ref_dates[has_ref] - issue_dates[has_ref]).astype("timedelta64[D]").astype(int)
        from_ref[has_ref] = np.searchsorted(days, ref_offsets - 1, side="right") + 1
        from_ref[from_ref > len(days)] = 9999
    return fixed.astype(int), from_ref


def _file_context(path: Path, context: dict) -> dict:
    """The context for one file, read in its own format: its dimension names
    and the scale that puts its rainfall in millimetres (see formats)."""
    fmt = io.of_file(path, context["variable"], context["dims"], context["fallback_unit_scale"])
    dimensions = context["spec"]["dimensions"]
    spec = {**context["spec"], "dimensions": {**dimensions, "rename": {**dimensions["rename"], **fmt.rename}}}
    return {**context, "spec": spec, "unit_scale": fmt.scale}


def _forecast_file(path: Path, context: dict) -> tuple[xr.Dataset, bool]:
    """One forecast file's onset counts and rain summaries, a run of issues
    at a time."""
    context = _file_context(path, context)
    points = io.forecast_points(path, context["variable"], context["spec"]["dimensions"]["rename"])
    if points is None:
        raise ValueError(
            f"{path.name}: no {context['variable']!r} variable over issue time, lead day, optional member and "
            "spatial dimensions only"
        )
    if context["cell_transform"] is not None:
        transform, source = _cell_transform(points, context["cell_transform"], context["convention"], path.name)
        return _forecast_onto_units(path, transform, source, transform.target_ids, context, scale_first=True)
    if context["regridder"] is not None and "id" not in points.columns:
        regridder = context["regridder"]
        transform = regridder.transform(*domain.grid_axes(points))
        source = domain.point_sources(points, transform)
        return _forecast_onto_units(path, transform, source, regridder.targets, context, scale_first=False)
    points, keep, in_area = _kept_mask(points, context, path)
    spec = context["spec"]
    with io.ForecastFile(
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


def _forecast_onto_units(path: Path, transform, source: np.ndarray, targets, context: dict, *,
                         scale_first: bool) -> tuple[xr.Dataset, bool]:
    """A forecast summarised on other units (a regrid or a cell transform):
    each run of issues is read at the points feeding the kept units and
    transformed in memory. A regrid scales after (the legacy order: regrid
    the file, scale as it is read), a cell transform before."""
    units = pd.DataFrame({"id": pd.Series(np.asarray(targets, dtype=object), dtype=str)})
    in_area = _inside(context["area"], units["id"])
    keep = in_area & (units["id"].isin(context["keep_ids"]).to_numpy() if context["keep_ids"] is not None else True)
    kept_units = np.flatnonzero(keep)
    feeding = np.isin(source, domain.Regridder.sources_of(transform, kept_units))
    spec = context["spec"]
    scale = context["unit_scale"]
    with io.ForecastFile(path, context["variable"], spec["dimensions"]["rename"], spec, feeding,
                         scale=scale if scale_first else 1.0) as forecast:
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
            if not scale_first and scale != 1.0:
                regridded = regridded * scale
            if forecast.members is not None and not member_spread:
                member_spread = bool(((~np.isnan(regridded)).sum(axis=1) >= 2).any())
            summary.add(start, stop, np.ascontiguousarray(regridded[:, selected]))
    return summary.dataset(), member_spread


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
        fixed, from_ref = _start_days(
            issue_rows, self.ref.reshape(-1), np.asarray(days), context["spec"]["options"]["fixed_cutoff_month_day"],
        )
        self.fixed, self.from_ref = fixed.reshape(n_issue, n_unit), from_ref.reshape(n_issue, n_unit)
        self.counts = {v: np.zeros((n_issue, n_unit, n_day), dtype=np.uint8) for v in VARIANTS}
        self.members = np.zeros(n_issue, dtype=np.uint8)
        self.rain_mean = None

    def add(self, start: int, stop: int, rain: np.ndarray) -> None:
        rule = self.context["onset_rule"]
        n_run, n_member, n_unit, n_day = rain.shape
        units = np.tile(np.arange(n_unit), n_member)
        thresholds = np.tile(self.thresholds, n_member)
        for offset in range(n_run):
            t = start + offset
            starts = np.vstack((
                np.ones(n_member * n_unit, dtype=int), np.tile(self.fixed[t], n_member),
                np.tile(self.from_ref[t], n_member),
            ))
            found = find_onsets(rain[offset].reshape(n_member * n_unit, n_day), thresholds, starts, rule)
            for variant, onset in zip(VARIANTS, found):
                valid = np.isfinite(onset) & (onset >= 1) & (onset <= n_day)
                np.add.at(self.counts[variant][t], (units[valid], onset[valid].astype(np.intp) - 1), 1)
        self.members[start:stop] = n_member

        if self.rain_mean is None:
            self.rain_mean = np.empty((len(self.issues), n_unit, n_day), dtype=rain.dtype)
        # One member is its own mean, with no spread.
        self.rain_mean[start:stop] = rain[:, 0] if n_member == 1 else member_mean(rain)

    def dataset(self) -> xr.Dataset:
        variables = {
            "rain_mean": (("issue", "unit", "day"), self.rain_mean),
            "members": (("issue",), self.members),
            "onset_threshold": (("unit",), self.thresholds),
            "year": (("issue",), pd.DatetimeIndex(self.issues).year.to_numpy(dtype=np.int64)),
            **{f"count_{v}": (("issue", "unit", "day"), self.counts[v]) for v in VARIANTS},
        }
        return xr.Dataset(
            variables, coords={"issue": self.issues, "unit": self.ids.astype(object), "day": np.asarray(self.days)}
        )


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
        points = io.forecast_points(path, context["variable"], context["spec"]["dimensions"]["rename"])
        if points is not None and len(points):
            kept = float(domain.with_ids(points, context["convention"])["id"].isin(context["keep_ids"]).mean())
    run_values = values * kept * min(1.0, run / max(n_times, 1))
    return int(run_values * _FORECAST_BYTES_PER_RUN_VALUE) + _FORECAST_FIXED_BYTES


def forecast(
    config: BlendConfig,
    model: ForecastModel,
    *,
    keep_ids: frozenset[str] | None = None,
    reference: Reference | None = None,
    area: domain.AreaFilter | None = None,
    member_ids: list[int] | None = None,
    workers: int = 1,
    budget_bytes: int | None = None,
) -> ForecastOnsets:
    """A model's forecast onsets for every one of its files.

    keep_ids, when given, limits the units to those (the blendable units
    under the inner join); the results for them are unchanged. reference and area
    default to the config's (a forecast passes the trained blend's);
    member_ids limits the members read (a forecast reads the trained ones).
    """
    paths = io.list_files(model.files, config, f"model {model.name}")
    spec = _specs.forecast_spec(config, model)
    context = {
        **_context(config, spec, reference or Reference.resolve(config), model.name, area),
        "variable": model.variable,
        # Each file's own dimension names and units decide how it is read;
        # these apply where a file does not say (see _file_context).
        "dims": dict(model.dims),
        "fallback_unit_scale": float(model.unit_scale),
        "keep_ids": keep_ids,
        "max_member": model.max_member,
        "member_ids": None if member_ids is None else [int(m) for m in member_ids],
    }
    count = parallel.workers_within([_forecast_bytes(p, context) for p in paths], workers, budget_bytes)
    results = parallel.map_ordered(_forecast_file, paths, context, count)
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
