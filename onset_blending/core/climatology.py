"""The KDE climatology of onset dates: conditional and unconditional onset
probabilities by issue date and lead day, plus windowed variants.

A unit's probabilities depend on the issue date only through its day of the
season (and the horizon that day has), so they are computed once per
distinct (season day, horizon), or per (year, season day, horizon) when the
KDE is refitted without each test year, instead of once per issue date of
every test year.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from . import parallel
from .._legacy import climatology_utils as clim_utils
from ..config import BlendConfig

# Units per process task: enough to amortise a task, few enough to spread.
_UNITS_PER_TASK = 200


@dataclass
class Climatology:
    """Onset probabilities by unit, season day and lead day.

    data holds conditional (leads 1..H, NaN at lead 0) and unconditional
    (leads 0..H) over (unit, season_day, lead), with a year dimension when
    the KDE is refitted without each test year (or a season day's horizon
    differs by year); its issue coordinate lists the test years' issue
    dates with their year and season day. windows holds conditional
    probabilities trained on other year windows, each over its own units."""

    data: xr.Dataset | None
    train_years: tuple[int, int]
    test_years: tuple[int, int]
    eligible_units: int
    windows: dict[tuple[int, int], xr.DataArray] = field(default_factory=dict)

    @property
    def by_year(self) -> bool:
        return "year" in self.data["conditional"].dims

    def release(self) -> None:
        """Drop the arrays, keeping the year windows."""
        self.data = None
        self.windows = {}

    def at(self, name: str, unit, year, season_day, leads: slice | None = None) -> np.ndarray:
        """Probabilities (rows, lead) for rows given as unit positions, years
        and season days; leads selects lead days (positions) when given."""
        array = self.data[name] if leads is None else self.data[name].isel(lead=leads)
        return _gather(array, unit, year, season_day)

    def legacy_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The (conditional, unconditional) tables the legacy pipeline wrote."""
        return (
            _legacy_table(self.data["conditional"].isel(lead=slice(1, None)), self.data, True),
            _legacy_table(self.data["unconditional"], self.data, False),
        )

    def legacy_window_table(self, window: tuple[int, int]) -> pd.DataFrame:
        return _legacy_table(self.windows[window].isel(lead=slice(1, None)), self.data, True)


def _gather(array: xr.DataArray, unit, year, season_day) -> np.ndarray:
    day = np.searchsorted(array["season_day"].values, season_day)
    if "year" in array.dims:
        return array.values[unit, np.searchsorted(array["year"].values, year), day]
    return array.values[unit, day]


def _legacy_table(array: xr.DataArray, data: xr.Dataset, conditional: bool) -> pd.DataFrame:
    cv = bool(data.attrs["cv_by_year"])
    model = ("clim_kde" if conditional else "clim_kde_unc") + ("_cv" if cv else "")
    units = array["unit"].values
    n_issue = data.sizes["issue"]
    unit_rows = np.repeat(np.arange(len(units)), n_issue)
    issue_rows = np.tile(np.arange(n_issue), len(units))
    years = data["issue_year"].values[issue_rows]
    columns = {
        "time": pd.Series(np.tile(pd.DatetimeIndex(data["issue"].values).date, len(units)), dtype=object),
        "year": years.astype(np.int64),
        "id": pd.Series(units[unit_rows], dtype=str),
        "model": pd.Series(np.full(len(unit_rows), model, dtype=object), dtype=str),
    }
    values = _gather(array, unit_rows, years, data["issue_season_day"].values[issue_rows])
    leads = [f"predicted_prob_day_{d}" for d in array["lead"].values]
    return pd.concat([pd.DataFrame(columns), pd.DataFrame(values, columns=leads, copy=False)], axis=1)


def years(
    config: BlendConfig,
    observed_years: list[int],
    forecast_years: list[int],
    *,
    test_years: tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """(train, test) year windows: the configured ones, else the test window
    spans the forecast years and training runs from the first observed year
    to the year before the first test year."""
    if not observed_years:
        raise ValueError("Cannot build a climatology without observed years")
    if not forecast_years:
        raise ValueError("Cannot build a climatology without forecast years")
    test = test_years or config.climatology.test_years or (min(forecast_years), max(forecast_years))
    configured = config.climatology.train_years
    if configured is not None:
        return tuple(configured), tuple(test)
    first = min(observed_years)
    last = min(test[0] - 1, max(observed_years))
    if last < first:
        last = max(observed_years)
    return (first, last), tuple(test)


def _training_onsets(observed, window: tuple[int, int], min_onset_years: int):
    onsets = clim_utils.filter_gt_training(observed.onsets(), *window)
    counts = onsets.groupby("id")["onset_day"].size()
    eligible = counts[counts >= int(min_onset_years)].index
    onsets = onsets[onsets["id"].isin(eligible)].copy()
    if onsets.empty:
        raise ValueError(
            "No units have enough observed onset years for a climatology "
            f"(min_onset_years={min_onset_years}, years {window[0]}:{window[1]})"
        )
    return onsets, len(eligible)


def _plan(config: BlendConfig, test: tuple[int, int], cv_by_year: bool):
    """The test years' issue grid (time, year, d0, window), whether the
    probabilities vary by year, and the longest horizon."""
    settings = config.climatology
    horizons = [
        {"start_md": h.start, "end_md": h.end, "forecast_window": h.forecast_window}
        for h in settings.horizons
    ] or None
    window_days = None if horizons else config.climatology_forecast_window()
    grid = clim_utils.build_issue_grid(test[0], test[1], config.onset.season_start, settings.issue_end)
    plan = clim_utils._build_issue_plan(grid, config.onset.season_start, window_days, horizons)
    grid = pd.DataFrame({"time": plan["time"], "year": plan["year"], "d0": plan["d0"], "window": plan["window"]})
    by_year = cv_by_year or bool((grid.groupby("d0")["window"].nunique() > 1).any())
    return grid, by_year, plan["max_h"]


def _climatology_task(task: tuple, context: dict) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities for a run of units at each plan key: (unit, key, lead) arrays."""
    years_by_unit, days_by_unit = task
    keys, max_h, cv = context["keys"], context["max_h"], context["cv_by_year"]
    d0 = keys["d0"].to_numpy(dtype=int)
    windows = keys["window"].to_numpy(dtype=int)
    conditional = np.full((len(days_by_unit), len(keys), max_h), np.nan)
    unconditional = np.full((len(days_by_unit), len(keys), max_h + 1), np.nan)
    for u, (unit_years, onset_days) in enumerate(zip(years_by_unit, days_by_unit)):
        if not cv:
            conditional[u], unconditional[u] = clim_utils._predict_pair_for_rows(
                clim_utils.fit_kde(onset_days), d0, windows, max_h
            )
            continue
        key_years = keys["year"].to_numpy(dtype=int)
        for year in np.unique(key_years):
            rows = np.flatnonzero(key_years == year)
            kde = clim_utils.fit_kde(onset_days[unit_years != year])
            conditional[u, rows], unconditional[u, rows] = clim_utils._predict_pair_for_rows(
                kde, d0[rows], windows[rows], max_h
            )
    clim_utils._kde_cdf.cache_clear()
    return conditional, unconditional


def _probabilities(onsets: pd.DataFrame, grid: pd.DataFrame, by_year: bool, max_h: int,
                   cv_by_year: bool, workers: int) -> xr.Dataset:
    """conditional and unconditional over (unit, [year,] season_day, lead) for
    every unit of the training onsets, in their order of appearance."""
    columns = ["year", "d0", "window"] if by_year else ["d0", "window"]
    keys = grid[columns].drop_duplicates().reset_index(drop=True)
    context = {"keys": keys, "max_h": max_h, "cv_by_year": cv_by_year}
    units = onsets["id"].unique()
    grouped = onsets.groupby("id", sort=False)
    per_unit = [grouped.get_group(u) for u in units]
    tasks = [
        (
            [g["year"].to_numpy(dtype=int) for g in per_unit[start:start + _UNITS_PER_TASK]],
            [g["onset_day"].to_numpy(dtype=float) for g in per_unit[start:start + _UNITS_PER_TASK]],
        )
        for start in range(0, len(per_unit), _UNITS_PER_TASK)
    ]
    results = parallel.map_ordered(_climatology_task, tasks, context, workers)
    conditional = np.concatenate([r[0] for r in results])
    unconditional = np.concatenate([r[1] for r in results])

    season_days = np.unique(keys["d0"].to_numpy())
    day = np.searchsorted(season_days, keys["d0"].to_numpy())
    leads = np.arange(max_h + 1)
    if by_year:
        years = np.unique(keys["year"].to_numpy())
        year = np.searchsorted(years, keys["year"].to_numpy())
        shape = (len(units), len(years), len(season_days), len(leads))
        index = (slice(None), year, day)
        dims, coords = ("unit", "year", "season_day", "lead"), {"year": years}
    else:
        shape = (len(units), len(season_days), len(leads))
        index = (slice(None), day)
        dims, coords = ("unit", "season_day", "lead"), {}
    dense_conditional = np.full(shape, np.nan)
    dense_unconditional = np.full(shape, np.nan)
    dense_conditional[index + (slice(1, None),)] = conditional
    dense_unconditional[index] = unconditional
    return xr.Dataset(
        {"conditional": (dims, dense_conditional), "unconditional": (dims, dense_unconditional)},
        coords={"unit": np.asarray(units, dtype=object), "season_day": season_days, "lead": leads, **coords},
    )


def build(
    config: BlendConfig,
    observed,
    forecast_years: list[int],
    *,
    test_years: tuple[int, int] | None = None,
    workers: int = 1,
) -> Climatology:
    """The climatologies for the issue dates of the test years.

    test_years overrides the config's (a live season scores one year)."""
    settings = config.climatology
    observed_years = observed.years
    train, test = years(config, observed_years, forecast_years, test_years=test_years)
    grid, by_year, max_h = _plan(config, test, settings.cv_by_year)
    workers = settings.workers or workers

    onsets, eligible = _training_onsets(observed, train, settings.min_onset_years)
    data = _probabilities(onsets, grid, by_year, max_h, settings.cv_by_year, workers)
    data = data.assign_coords(
        issue=pd.to_datetime(grid["time"]).to_numpy(),
        issue_year=("issue", grid["year"].to_numpy(dtype=np.int64)),
        issue_season_day=("issue", grid["d0"].to_numpy(dtype=np.int64)),
    )
    data.attrs.update(cv_by_year=int(settings.cv_by_year))
    windows = {}
    for window in settings.windows:
        window_onsets, _ = _training_onsets(observed, window, settings.min_onset_years)
        windows[window] = _probabilities(
            window_onsets, grid, by_year, max_h, settings.cv_by_year, workers
        )["conditional"]
    return Climatology(
        data=data,
        train_years=train,
        test_years=test,
        eligible_units=eligible,
        windows=windows,
    )
