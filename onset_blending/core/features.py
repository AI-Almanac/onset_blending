"""Features: the weekly features the blend trains on, built from the stage arrays.

A sample is a unit and issue date that the climatology and every model
cover (the inner join of today's combine), in the climatology's order
(units, then issue dates), dropping issue dates after the observed onset.
Its weekly features (climatology logits, binned forecast probabilities,
rain predictors, the outcome) are computed from the arrays directly and
held as arrays (`Weekly`); the legacy table is `Weekly.frame()`. The
random forest reads its day-level inputs as arrays too (`daily_inputs`).

The numerics are the legacy connector's (week sums, winsorised logits,
rolling rain sums over the week's start days, rain transforms), operation
for operation, without its day-level table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from scipy.special import logit

from .. import config as cfg
from .climatology import Climatology
from ..config import BlendConfig
from .onsets import ForecastOnsets, ObservedOnsets
from .predictions import outcome_codes, outcome_series

_CHUNK_ROWS = 1_000_000


@dataclass
class Rows:
    """The feature rows: unit and issue positions into each source."""

    units: np.ndarray        # unit ids, per row
    times: np.ndarray        # issue dates, datetime64[ns], per row
    years: np.ndarray        # per row
    season_day: np.ndarray   # per row
    clim_unit: np.ndarray    # position in the climatology's units
    model_unit: dict[str, np.ndarray]
    model_issue: dict[str, np.ndarray]
    true_onset: np.ndarray   # datetime64[D], NaT where there is none
    true_onset_day: np.ndarray

    def __len__(self) -> int:
        return len(self.units)


def rows(config: BlendConfig, climatology: Climatology, observed: ObservedOnsets,
         forecasts: dict[str, ForecastOnsets], *, trim: bool | None = None) -> Rows:
    """The units and issue dates every source covers, in the climatology's order."""
    clim = climatology.data
    clim_units = clim["unit"].values.astype(str)
    unit_ok = np.ones(len(clim_units), dtype=bool)
    issue_ok = np.ones(clim.sizes["issue"], dtype=bool)
    issues = clim["issue"].values
    for onsets in forecasts.values():
        unit_ok &= np.isin(clim_units, onsets.data["unit"].values.astype(str))
        issue_ok &= np.isin(issues, onsets.data["issue"].values)
    unit_pos = np.flatnonzero(unit_ok)
    issue_pos = np.flatnonzero(issue_ok)
    clim_unit = np.repeat(unit_pos, len(issue_pos))
    grid = np.tile(issue_pos, len(unit_pos))
    units = clim_units[clim_unit]
    times = issues[grid]
    years = clim["issue_year"].values[grid]

    observed_units = observed.data["unit"].values.astype(str)
    order = np.argsort(observed_units)
    found = order[np.searchsorted(observed_units, units, sorter=order).clip(0, len(order) - 1)]
    has_unit = observed_units[found] == units
    year_pos = np.searchsorted(observed.data["year"].values, years).clip(0, observed.data.sizes["year"] - 1)
    has_year = observed.data["year"].values[year_pos] == years
    known = has_unit & has_year
    onset_dates = observed.onset_dates()
    true_onset = np.where(known, onset_dates[found, year_pos], np.datetime64("NaT"))
    true_onset_day = np.where(known, observed.data["onset_day"].values[found, year_pos], np.nan)

    if trim if trim is not None else config.features.trim_after_true_onset:
        day = times.astype("datetime64[D]")
        keep = np.isnat(true_onset) | (day <= true_onset)
    else:
        keep = np.ones(len(units), dtype=bool)

    model_unit, model_issue = {}, {}
    for name, onsets in forecasts.items():
        m_units = onsets.data["unit"].values.astype(str)
        m_order = np.argsort(m_units)
        model_unit[name] = m_order[np.searchsorted(m_units, units[keep], sorter=m_order)]
        model_issue[name] = np.searchsorted(onsets.data["issue"].values, times[keep])
    return Rows(
        units=units[keep],
        times=times[keep],
        years=years[keep],
        season_day=clim["issue_season_day"].values[grid][keep],
        clim_unit=clim_unit[keep],
        model_unit=model_unit,
        model_issue=model_issue,
        true_onset=true_onset[keep],
        true_onset_day=true_onset_day[keep],
    )


def _week_sums(daily: np.ndarray, days_per_bin: int, n_bins: int) -> list[np.ndarray]:
    """Weekly sums of daily values (rows, days 1..D), as the legacy connector sums them."""
    daily = np.asfortranarray(daily)
    return [daily[:, (w - 1) * days_per_bin : w * days_per_bin].sum(axis=1) for w in range(1, n_bins + 1)]


def _outcome_codes(lead: np.ndarray, days_per_bin: int, n_bins: int) -> tuple[np.ndarray, list[str]]:
    """assign_lead_bin for every row, as positions into the labels (week<w>,
    later), -1 where it is missing."""
    labels = [f"week{w}" for w in range(1, n_bins + 1)] + ["later"]
    codes = np.full(len(lead), -1, dtype=np.int8)
    remaining = ~np.isnan(lead) & (lead > 0)
    for w in range(1, n_bins + 1):
        hit = remaining & (lead <= w * days_per_bin)
        codes[hit] = w - 1
        remaining &= ~hit
    codes[remaining] = n_bins
    return codes, labels


# Names a Weekly uses for its dimensions and bookkeeping.
_RESERVED = ("sample", "unit", "sample_unit")


@dataclass
class Weekly:
    """The weekly features as arrays: one sample per unit and issue date.

    `data` holds each feature over `sample`, with the sample's position in
    the `unit` dimension (sample_unit), its issue time and year and its
    outcome (positions into the outcome labels, -1 without one); the unit
    ids and the per-unit columns (lat, lon, onset_threshold) are over
    `unit`. A column reads as an array per sample (`weekly["diff_x_week1"]`)
    and `frame()` builds the legacy table from them, for export and the
    code that still reads tables. `from_frame` takes a legacy table,
    keeping every column as it is."""

    data: xr.Dataset
    columns: list[str]                   # the legacy table's columns, in order
    attrs: dict = field(default_factory=dict)
    dtypes: dict = field(default_factory=dict)   # legacy dtypes, for tables taken by from_frame

    def __len__(self) -> int:
        return self.data.sizes["sample"]

    def __contains__(self, name) -> bool:
        return name in self.columns

    def __getitem__(self, name: str) -> np.ndarray:
        return self.take(name)

    @property
    def units(self) -> np.ndarray:
        return self.data["unit"].values

    @property
    def sample_unit(self) -> np.ndarray:
        return self.data["sample_unit"].values

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    def take(self, name: str, rows: np.ndarray | None = None) -> np.ndarray:
        """A column's values per sample, for `rows` (positions) or every sample."""
        if name not in self.columns:
            raise KeyError(name)
        unit = self.sample_unit if rows is None else self.sample_unit[rows]
        if name == "id":
            return self.units[unit]
        variable = self.data[name]
        if variable.dims == ("unit",):
            return variable.values[unit]
        values = variable.values if rows is None else variable.values[rows]
        if "labels" in variable.attrs:
            return outcome_series(values, list(variable.attrs["labels"])).to_numpy()
        return values

    def column_dtypes(self) -> dict[str, np.dtype]:
        """Each column's dtype as `take` returns it (labels are objects)."""
        out = {}
        for name in self.columns:
            if name == "id" or "labels" in self.data[name].attrs:
                out[name] = np.dtype(object)
            else:
                out[name] = self.data[name].dtype
        return out

    def outcome_codes(self, rows: np.ndarray | None = None, labels=None) -> np.ndarray:
        """The samples' outcomes as positions into `labels` (default: the
        stored labels): -1 without one, len(labels) for another label."""
        variable = self.data["outcome"]
        if "labels" not in variable.attrs:
            return outcome_codes(self.take("outcome", rows), labels)
        codes = variable.values if rows is None else variable.values[rows]
        stored = list(variable.attrs["labels"])
        if labels is None or list(labels) == stored:
            return codes.astype(np.int8)
        index = {label: k for k, label in enumerate(labels)}
        remap = np.array([index.get(label, len(labels)) for label in stored] + [-1], dtype=np.int8)
        return remap[codes]

    def frame(self, rows: np.ndarray | None = None, columns=None) -> pd.DataFrame:
        """The legacy table, or some rows (positions) and columns of it, on a
        fresh 0..n-1 index."""
        wanted = self.columns if columns is None else [c for c in self.columns if c in set(columns)]
        out = {name: self._series(name, rows) for name in wanted}
        table = pd.DataFrame(out) if out else pd.DataFrame(index=pd.RangeIndex(len(self) if rows is None else len(rows)))
        table.attrs.update(self.attrs)
        return table

    def _series(self, name: str, rows):
        if name in self.dtypes:
            return pd.Series(self.take(name, rows), dtype=self.dtypes[name])
        variable = None if name == "id" else self.data[name]
        if variable is not None and "labels" in variable.attrs:
            codes = variable.values if rows is None else variable.values[rows]
            return outcome_series(codes, list(variable.attrs["labels"]))
        values = self.take(name, rows)
        if name == "id":
            return pd.Series(values, dtype=str)
        if name == "time":
            return pd.Series(pd.DatetimeIndex(values).date, dtype=object)
        return values

    def to_netcdf(self, path) -> None:
        """The features as NetCDF, for `Weekly.open` (features built from
        the arrays; a table taken by from_frame keeps its own dtypes)."""
        if self.dtypes:
            raise ValueError("a Weekly taken from a table is saved as that table")
        data = self.data.copy()
        data.attrs = {"columns": json.dumps(self.columns), "attrs": json.dumps(self.attrs, default=_json_scalar)}
        for name in data.variables:
            if "labels" in data[name].attrs:
                data[name].attrs["labels"] = json.dumps(list(data[name].attrs["labels"]))
        data.to_netcdf(path)

    @classmethod
    def open(cls, path) -> "Weekly":
        with xr.open_dataset(path) as stored:
            data = stored.load()
        columns, attrs = json.loads(data.attrs.pop("columns")), json.loads(data.attrs.pop("attrs"))
        for name in data.variables:
            if "labels" in data[name].attrs:
                data[name].attrs["labels"] = json.loads(data[name].attrs["labels"])
        data = data.assign_coords(unit=np.asarray(data["unit"].values, dtype=object))
        return cls(data=data, columns=columns, attrs=attrs)

    def with_units(self, ids: np.ndarray) -> "Weekly":
        """The same samples with these unit ids (one per unit)."""
        data = self.data.assign_coords(unit=np.asarray(ids, dtype=object))
        return Weekly(data=data, columns=self.columns, attrs=self.attrs, dtypes=self.dtypes)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Weekly":
        """A legacy table: its ids as units, every other column as it is."""
        if "id" not in frame.columns:
            raise ValueError("the weekly table has no id column")
        reserved = [c for c in frame.columns if c in _RESERVED]
        if reserved:
            raise ValueError(f"weekly table columns {reserved} clash with the Weekly layout")
        codes, units = pd.factorize(frame["id"], use_na_sentinel=False)
        variables = {"sample_unit": ("sample", codes.astype(np.int32))}
        for name in frame.columns:
            if name != "id":
                variables[name] = ("sample", frame[name].to_numpy())
        data = xr.Dataset(variables, coords={"unit": np.asarray(units, dtype=object)})
        return cls(
            data=data,
            columns=[str(c) for c in frame.columns],
            attrs=dict(frame.attrs),
            dtypes={name: frame[name].dtype for name in frame.columns},
        )

    @classmethod
    def of(cls, table) -> "Weekly":
        return table if isinstance(table, cls) else cls.from_frame(table)


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _grid_lat_lon(units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lat/lon parsed from "<lat>_<lon>" ids (NaN otherwise), per distinct id."""
    codes, distinct = pd.factorize(units)
    text = pd.Series(distinct, dtype=str)
    parts = text.str.split("_", n=1, expand=True)
    lat = pd.to_numeric(parts[0], errors="coerce")
    lon = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else pd.Series(np.nan, index=text.index)
    grid = (text.str.count("_") == 1) & lat.between(-90, 90) & lon.between(-180, 360)
    return lat.where(grid).to_numpy()[codes], lon.where(grid).to_numpy()[codes]


def weekly(config: BlendConfig, climatology: Climatology, observed: ObservedOnsets,
           forecasts: dict[str, ForecastOnsets], *, trim: bool | None = None) -> Weekly:
    """The weekly features the legacy connector built from the day-level table.

    trim overrides features.trim_after_true_onset (a forecast never trims)."""
    dpb, n_bins = config.features.bins.days, config.features.bins.count
    r = rows(config, climatology, observed, forecasts, trim=trim)
    first = config.models[0].name
    threshold = forecasts[first].data["onset_threshold"].values[r.model_unit[first]]
    units, sample_unit = np.unique(r.units, return_inverse=True)
    lat, lon = _grid_lat_lon(units)
    unit_threshold = np.full(len(units), np.nan)
    unit_threshold[sample_unit] = threshold  # one model unit per id
    lead = (r.true_onset - r.times.astype("datetime64[D]")).astype("timedelta64[D]").astype(float)
    lead[np.isnat(r.true_onset)] = np.nan
    codes, labels = _outcome_codes(lead, dpb, n_bins)

    variables: dict[str, tuple] = {
        "sample_unit": ("sample", sample_unit.astype(np.int32)),
        "time": ("sample", r.times),
        "year": ("sample", pd.DatetimeIndex(r.times).year.to_numpy()),
        "lat": ("unit", lat),
        "lon": ("unit", lon),
        "onset_threshold": ("unit", unit_threshold),
        "outcome": ("sample", codes, {"labels": labels}),
    }
    # Every feature of a row comes from that row alone, so they are built a
    # chunk of rows at a time into preallocated columns, which bounds the
    # working arrays.
    attrs = {"rain_horizons": {}, "unavailable_rain_predictors": {}}
    features: dict[str, np.ndarray] = {}
    for start in range(0, max(len(r), 1), _CHUNK_ROWS):
        rows_here = slice(start, min(len(r), start + _CHUNK_ROWS))
        for name, values in _row_features(config, climatology, forecasts, r, rows_here, threshold, attrs).items():
            if name not in features:
                features[name] = np.empty(len(r), dtype=values.dtype)
            features[name][rows_here] = values
    for name, values in features.items():
        variables[name] = ("sample", values)
    data = xr.Dataset(variables, coords={"unit": units.astype(object)})
    columns = ["id", "time", "year", "lat", "lon", "onset_threshold", "outcome", *features]
    return Weekly(data=data, columns=columns, attrs=attrs)


def _row_features(config, climatology, forecasts, r: Rows, rows_here: slice, threshold, attrs) -> dict:
    """The weekly features of a slice of rows, in the legacy column order."""
    features = config.features
    day_max, dpb, n_bins = features.probability_days, features.bins.days, features.bins.count
    unit, years, season_day = r.clim_unit[rows_here], r.years[rows_here], r.season_day[rows_here]
    out = {}

    feature = cfg.CLIMATOLOGY_FEATURE
    cond = climatology.at("conditional", unit, years, season_day)
    unc = climatology.at("unconditional", unit, years, season_day)
    cond_weeks = _week_sums(cond[:, 1 : day_max + 1], dpb, n_bins)
    unc_weeks = _week_sums(unc[:, 1 : day_max + 1], dpb, n_bins)
    out[f"{feature}_unc_earlier"] = _logit_winsor(unc[:, 0].copy())
    for w in range(n_bins):
        out[f"{feature}_week{w + 1}"] = _logit_winsor(cond_weeks[w])
        out[f"{feature}_unc_week{w + 1}"] = _logit_winsor(unc_weeks[w])
    del cond, unc
    ids = r.units[rows_here]
    for window in config.climatology.windows:
        tag = cfg.window_tag(window)
        array = climatology.windows[window]
        units = array["unit"].values.astype(str)
        order = np.argsort(units)
        position = order[np.searchsorted(units, ids, sorter=order).clip(0, len(order) - 1)]
        values = _windowed_at(array, position, years, season_day)
        values[units[position] != ids] = np.nan
        for w, week in enumerate(_week_sums(values[:, 1 : day_max + 1], dpb, n_bins), start=1):
            out[f"{feature}_{tag}_week{w}"] = _logit_winsor(week)

    # The onset series each probability variant counts: base, then those read.
    series = {"base": "base", "fixed_cutoff": "fixed_cutoff", "ref_onset": "ref"}
    variants = ["base", *cfg.probability_variants(config)]
    for model in config.models:
        data = forecasts[model.name].data
        issue, unit_m = r.model_issue[model.name][rows_here], r.model_unit[model.name][rows_here]
        members = data["members"].values[issue].astype(float)[:, None]
        for variant in variants:
            daily_probs = data[f"count_{series[variant]}"].values[issue, unit_m, :day_max] / members
            prefix = cfg.probability_feature_base(model.name, variant)
            for w, week in enumerate(_week_sums(daily_probs, dpb, n_bins), start=1):
                out[f"{prefix}_week{w}"] = week

    out.update(_rain_predictors(config, forecasts, r, rows_here, threshold[rows_here], attrs))
    return out


def _windowed_at(array, unit, year, season_day) -> np.ndarray:
    day = np.searchsorted(array["season_day"].values, season_day)
    if "year" in array.dims:
        return array.values[unit, np.searchsorted(array["year"].values, year), day]
    return array.values[unit, day]


def _rain_predictors(config: BlendConfig, forecasts: dict[str, ForecastOnsets], r: Rows, rows_here: slice,
                     threshold: np.ndarray, attrs: dict) -> dict:
    """Weekly maxima (or minima) of rolling rain sums, as the legacy connector computes them."""
    features = config.features
    day_max, dpb, n_bins = features.probability_days, features.bins.days, features.bins.count
    starts_by_week = [list(range((w - 1) * dpb + 1, w * dpb + 1)) for w in range(1, n_bins + 1)]
    out = {}
    for model in config.models:
        name = model.name
        if not model.rain_predictors:
            continue
        transform = rain_transform(model.rain_transform if model.rain_transform is not None else features.rain_transform)
        policy = model.rain_horizon.resolved_policy
        rain_day_max = model.rain_horizon.max_day if model.rain_horizon.max_day is not None else day_max + 10
        strict = policy != "legacy"
        rain = forecasts[name].data["rain_mean"]
        days = rain["day"].values
        need = [d for d in range(1, rain_day_max + 1)]
        missing = [d for d in need if d not in days]
        if missing:
            raise ValueError(f"{name}: rainfall lead days {missing[:5]} are missing for the rain predictors")
        positions = np.searchsorted(days, need)
        issue, unit = r.model_issue[name][rows_here], r.model_unit[name][rows_here]
        mat = np.asfortranarray(rain.values[issue, unit][:, positions].astype(float))
        if strict and not np.isfinite(mat).all():
            raise ValueError(f"{name}: non-finite rainfall within the strict rain horizon ({rain_day_max} days)")
        attrs["rain_horizons"][name] = {"rain_day_max": rain_day_max, "strict": strict, "policy": policy}
        parsed = [(p.agg, cfg.resolve_window(p, config.onset.definition)) for p in model.rain_predictors]
        rolled = {w: _rolling_sums(mat, w) for w in {w for _, w in parsed}}
        cache = {}
        for agg, window in parsed:
            for wi, starts in enumerate(starts_by_week):
                if (agg, window, wi) in cache:
                    continue
                valid = _complete_starts(rain_day_max, window, starts)
                cache[(agg, window, wi)] = _over_starts(rolled[window], valid, np.fmin if agg == "min" else np.fmax)
                if not valid:
                    column = f"diff_{name}_week{wi + 1}" if agg == "diff" else f"{agg}_{name}_{window}day_week{wi + 1}"
                    attrs["unavailable_rain_predictors"][column] = {
                        "model": name, "rain_day_max": rain_day_max, "window": window, "week": wi + 1,
                    }
        for agg, window in parsed:
            for w in range(1, n_bins + 1):
                values = cache[(agg, window, w - 1)]
                if agg == "diff":
                    out[f"diff_{name}_week{w}"] = transform(values) - transform(threshold)
                else:
                    out[f"{agg}_{name}_{window}day_week{w}"] = transform(values)
    return out


def _logit_winsor(p, lo: float = 0.0001, hi: float = 0.9999) -> np.ndarray:
    """The logit of a probability clipped to [lo, hi]."""
    return logit(np.clip(np.asarray(p, dtype=float), lo, hi))


def rain_transform(value):
    """The rainfall transform rain predictors apply (not onset detection or
    the climatology): identity, sqrt, fourth_root, log1p, power:<p> or a
    number p (x ** p). Powers and log1p clip at 0; NaN stays NaN."""
    def power(p):
        p = float(p)

        def f(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.isnan(x), x, np.clip(x, 0.0, None) ** p)
        return f

    def identity(x):
        return np.asarray(x, dtype=float)

    if value is None:
        return identity
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return power(value)
    name = str(value).strip().lower()
    if name in ("identity", "none", ""):
        return identity
    if name == "sqrt":
        return power(0.5)
    if name == "fourth_root":
        return power(0.25)
    if name == "log1p":
        def log1p(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.isnan(x), x, np.log1p(np.clip(x, 0.0, None)))
        return log1p
    if name.startswith("power:"):
        return power(name.split(":", 1)[1])
    raise ValueError(f"Unknown rain_transform {value!r}: identity, sqrt, fourth_root, log1p, 'power:<p>' or a number.")


def _rolling_sums(mat: np.ndarray, k: int) -> np.ndarray:
    """Rolling k-day sums along each row: (rows, days - k + 1)."""
    n = mat.shape[1]
    if n < k:
        return np.empty((mat.shape[0], 0))
    return np.stack([mat[:, s:s + k].sum(axis=1) for s in range(n - k + 1)], axis=1)


def _complete_starts(rain_day_max: int, window: int, starts) -> list[int]:
    """The week's start days whose whole rolling window lies within 1..rain_day_max."""
    last_start = int(rain_day_max) - int(window) + 1
    return [int(start) for start in starts if 1 <= int(start) <= last_start] if last_start >= 1 else []


def _over_starts(rolled: np.ndarray, starts, reduce) -> np.ndarray:
    """Per row, the max (np.fmax) or min (np.fmin) of the rolling sums at the
    start days (1-based), ignoring missing sums; NaN without any start."""
    ok = [s for s in starts if 1 <= s <= rolled.shape[1]]
    if rolled.shape[1] == 0 or not ok:
        return np.full(rolled.shape[0], np.nan)
    return reduce.reduce(rolled[:, [s - 1 for s in ok]], axis=1)


@dataclass
class DailyInputs:
    """What the random forest reads for each weekly sample, in the weekly
    samples' order: each model's ensemble-mean rainfall by lead day
    (`rain_<model>` over `<model>_day`), the conditional climatology's onset
    probability by lead day (`clim_p_onset` over `clim_day`), and the
    observed onset's lead day after the issue (`onset_lead`, NaN without
    one). `time` is each sample's issue date, to check the alignment."""

    data: xr.Dataset

    @property
    def model_names(self) -> list[str]:
        return [str(name)[len("rain_"):] for name in self.data.data_vars if str(name).startswith("rain_")]

    def rain(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """(lead days, rainfall (sample, day))."""
        variable = self.data[f"rain_{model}"]
        return variable[f"{model}_day"].values, variable.values

    @property
    def climatology(self) -> np.ndarray:
        return self.data["clim_p_onset"].values

    @property
    def onset_lead(self) -> np.ndarray:
        return self.data["onset_lead"].values


def daily_inputs(config: BlendConfig, climatology: Climatology, observed: ObservedOnsets,
                 forecasts: dict[str, ForecastOnsets], *, max_day: int, trim: bool | None = None) -> DailyInputs:
    """The random forest's inputs for the samples `weekly` builds (with the same trim)."""
    r = rows(config, climatology, observed, forecasts, trim=trim)
    lead = (r.true_onset - r.times.astype("datetime64[D]")).astype("timedelta64[D]").astype(float)
    lead[np.isnat(r.true_onset)] = np.nan
    clim = climatology.at("conditional", r.clim_unit, r.years, r.season_day, leads=slice(1, max_day + 1))
    variables: dict[str, tuple] = {
        "time": ("sample", r.times),
        "onset_lead": ("sample", lead),
        "clim_p_onset": (("sample", "clim_day"), clim),
    }
    coords = {"clim_day": np.arange(1, clim.shape[1] + 1)}
    for name, onsets in forecasts.items():
        rain = onsets.data["rain_mean"]
        variables[f"rain_{name}"] = (("sample", f"{name}_day"), rain.values[r.model_issue[name], r.model_unit[name]])
        coords[f"{name}_day"] = rain["day"].values
    return DailyInputs(xr.Dataset(variables, coords=coords))
