"""Operational forecasts: a trained blend applied to live forecast files.

Only the live files (and, optionally, the issue date) come from the caller.
The recipe, reference data and observed onsets all come from the
TrainedBlend, and the live files must have the structure the blend was
trained on; every difference is reported before any work starts.

Operationally each raw model gives one file holding one initialisation,
and that is the issue date forecast. Files holding a season of issues (a
replay) forecast every issue in it, or the one `issue_date` names.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .core import climatology as climatology_stage, domain, features, onsets
from .trained import issue_dates
from .config import FileSet

WEEKLY_FILENAME = "blended_forecast_probabilities.csv"
DAILY_FILENAME = "daily_onset_probabilities.csv"


class IncompatibleForecast(ValueError):
    """Live forecasts the trained blend cannot be applied to; lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("Cannot forecast with this blend:\n" + "\n".join(f"  - {p}" for p in self.problems))


@dataclass
class Forecast:
    """Blended weekly onset probabilities for the live issue dates."""

    weekly: pd.DataFrame        # id, time, onset_threshold, lat, lon, cv_week1..N, cv_later, clim_*
    features: pd.DataFrame      # the weekly feature rows the probabilities come from
    # The random forest blend's, when the blend has one: id, time, lat, lon,
    # p_day_1..D, p_later and their week bins cv_week1..N, cv_later.
    daily: pd.DataFrame | None = None

    def save(self, directory: str | Path) -> Path:
        """Write the forecast's files into a new directory, all or nothing."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            self.weekly.to_csv(staging / WEEKLY_FILENAME, index=False)
            if self.daily is not None:
                self.daily.to_csv(staging / DAILY_FILENAME, index=False)
            if directory.exists():
                directory.rmdir()
            os.replace(staging, directory)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return directory


def _as_paths(value) -> list[Path]:
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    return [Path(v) for v in value]


def _held_issues(blend, live: dict[str, list[Path]]) -> tuple[dict[str, list[dt.date]], list[str]]:
    """The issue dates each model's live files hold."""
    held, problems = {}, []
    for name, paths in live.items():
        try:
            model = blend.config.model(name)
        except (KeyError, ValueError):
            continue  # an unknown model: check_forecast_files reports it
        try:
            held[name] = sorted({d for p in paths for d in issue_dates(p, model)})
        except (OSError, ValueError, KeyError) as error:
            problems.append(f"{name}: cannot read the issue dates of its live files ({error})")
            continue
        if not held[name]:
            problems.append(f"{name}: the live files hold no issue dates")
    return held, problems


def _issue_problems(config: cfg.BlendConfig, issue_date: dt.date | None,
                    held: dict[str, list[dt.date]]) -> tuple[dt.date | None, int | None, list[str]]:
    """(issue date, season year, problems). With no issue date given, files
    holding one initialisation each forecast it; files holding more forecast
    every issue of their one season."""
    problems = []
    if issue_date is None and held and all(len(dates) == 1 for dates in held.values()):
        inits = {name: dates[0] for name, dates in held.items()}
        if len(set(inits.values())) == 1:
            issue_date = next(iter(inits.values()))
        else:
            listed = ", ".join(f"{name} {date}" for name, date in inits.items())
            problems.append(f"the live files are initialised on different dates ({listed}); every model needs the same one")
            return None, None, problems
    if issue_date is not None:
        year = issue_date.year
        start = dt.date.fromisoformat(f"{year}-{config.onset.season_start}")
        end = dt.date.fromisoformat(f"{year}-{config.climatology.issue_end}")
        if not start <= issue_date <= end:
            problems.append(f"issue date {issue_date} is outside the season ({start} to {end})")
        for name, dates in held.items():
            if dates and issue_date not in dates:
                nearby = [d.isoformat() for d in dates if abs((d - issue_date).days) <= 7]
                problems.append(
                    f"{name}: the live files have no {issue_date} issue"
                    + (f" (nearby: {', '.join(nearby)})" if nearby else "")
                )
        return issue_date, year, problems
    years = {d.year for dates in held.values() for d in dates}
    if len(years) != 1:
        problems.append("the live files must hold one initialisation per model, or one season (or give issue_date)")
        return None, None, problems
    return None, years.pop(), problems


def run(blend, forecasts: dict, *, issue_date=None, resources=None, fewer_members: bool = False) -> Forecast:
    """Blended probabilities for the live forecasts (model name -> file or files).
    fewer_members lets an ensemble's live files hold only some of the members
    the blend was trained on: its onset probabilities are shares of fewer
    members, so coarser and noisier than the blend's training saw."""
    from .training import Resources

    resources = resources or Resources.default()
    if isinstance(issue_date, str):
        issue_date = dt.date.fromisoformat(issue_date)
    live = {name: _as_paths(paths) for name, paths in forecasts.items()}
    problems = blend.check_forecast_files(live, fewer_members=fewer_members)
    held, read_problems = _held_issues(blend, live)
    issue_date, year, issue_problems = _issue_problems(blend.config, issue_date, held)
    problems += read_problems + issue_problems
    if problems:
        raise IncompatibleForecast(problems)

    config = dataclasses.replace(
        blend.config,
        models=tuple(
            dataclasses.replace(m, files=FileSet(paths=tuple(str(p.resolve()) for p in live[m.name])))
            for m in blend.config.models
        ),
    )
    # Only the trained units, with the trained reference data: no area,
    # dissemination or reference file is read.
    # A live file may hold more than the blend was trained on (longer leads,
    # more members, a larger domain): only the trained lead days, members and
    # units are read.
    keep = frozenset(blend.units)
    model_onsets = {
        model.name: onsets.forecast(
            config, model, keep_ids=keep, reference=blend.reference, area=domain.AreaFilter.everything(),
            member_ids=blend.signatures[model.name].get("members"),
            workers=resources.workers, budget_bytes=resources.budget_bytes,
        )
        for model in config.models
    }
    climatology = climatology_stage.build(
        config, blend.observed, [year], test_years=(year, year), workers=resources.workers
    )
    # No trimming: a forecast must not know when the onset came.
    weekly = features.weekly(config, climatology, blend.observed, model_onsets, trim=False)
    rows = np.arange(len(weekly))
    if issue_date is not None:
        rows = rows[weekly["time"].astype("datetime64[D]") == np.datetime64(issue_date, "D")]
    issue_rows = rows
    for column in _input_columns(blend.model, weekly):
        rows = rows[pd.notna(weekly.take(column, rows))]
    if not len(rows):
        raise IncompatibleForecast(["no unit and issue date has every feature the blend needs"])
    table = weekly.frame(rows)
    daily = None
    if blend.forest is not None:
        # Every sample of the issue dates: the forest imputes missing inputs.
        daily = _daily(blend, config, climatology, model_onsets, weekly, issue_rows)
    return Forecast(weekly=_probabilities(blend, table), features=table, daily=daily)


def _daily(blend, config, climatology, model_onsets, weekly, rows) -> pd.DataFrame:
    """The random forest blend's daily and weekly probabilities for `rows`."""
    from .blend import random_forest

    fitted = blend.forest
    inputs = features.daily_inputs(
        config, climatology, blend.observed, model_onsets, max_day=fitted.params.max_day, trim=False
    )
    models = list(config.model_names)
    names = random_forest.feature_names(inputs, models)
    if [names[i] if i < len(names) else None for i in fitted.columns] != list(fitted.features):
        raise IncompatibleForecast(["the live forecasts do not give the random forest's trained inputs"])
    probabilities = random_forest.predict(fitted, random_forest.feature_matrix(weekly, inputs, rows, models))
    out = weekly.frame(rows, ["id", "time", "lat", "lon"])
    for day in range(probabilities.shape[1] - 1):
        out[f"p_day_{day + 1}"] = probabilities[:, day]
    out["p_later"] = probabilities[:, -1]
    weeks = random_forest.weekly_from_daily(probabilities, fitted.params.days_per_week)
    for week, label in enumerate(cfg.outcome_labels(blend.config)):
        out[f"cv_{label}"] = weeks[:, week]
    return out


def _input_columns(model, weekly: features.Weekly) -> list[str]:
    """The weekly columns the fitted model's terms are made of."""
    names = {part for feature in model.features for part in str(feature).split(":")}
    return [c for c in weekly.columns if c in names]


def _probabilities(blend, rows: pd.DataFrame) -> pd.DataFrame:
    labels = cfg.outcome_labels(blend.config)
    predicted = blend.model.predict(rows)
    out = rows[["id", "time", "onset_threshold", "lat", "lon"]].copy()
    for label in labels:
        out[f"cv_{label}"] = predicted[f"cv_{label}"].to_numpy()
    # The climatology the blend starts from, as probabilities.
    weeks = [label for label in labels if label != "later"]
    clim = 1.0 / (1.0 + np.exp(-rows[[f"{cfg.CLIMATOLOGY_FEATURE}_{w}" for w in weeks]].to_numpy()))
    for index, week in enumerate(weeks):
        out[f"clim_{week}"] = clim[:, index]
    out["clim_later"] = np.maximum(0.0, 1.0 - clim.sum(axis=1))
    return out
