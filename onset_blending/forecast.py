"""Operational forecasts: a trained blend applied to live forecast files.

Only the live files and the issue date come from the caller. The recipe,
reference data and observed onsets all come from the TrainedBlend, and the
live files must have the structure the blend was trained on; every
difference is reported before any work starts.
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

from . import climatology as climatology_stage, config as cfg, domain, features, inputs, onsets
from .blend import issue_dates
from .config import FileSet

WEEKLY_FILENAME = "blended_forecast_probabilities.csv"


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

    def save(self, directory: str | Path) -> Path:
        """Write the forecast's files into a new directory, all or nothing."""
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} exists and is not empty")
        directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
        try:
            self.weekly.to_csv(staging / WEEKLY_FILENAME, index=False)
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


def _season_problems(config: cfg.BlendConfig, issue_date: dt.date | None, live: dict[str, list[Path]]) -> tuple[int | None, list[str]]:
    problems = []
    years = {inputs.file_year(p) for paths in live.values() for p in paths}
    if issue_date is not None:
        year = issue_date.year
        start = dt.date.fromisoformat(f"{year}-{config.onset.season_start}")
        end = dt.date.fromisoformat(f"{year}-{config.climatology.issue_end}")
        if not start <= issue_date <= end:
            problems.append(f"issue date {issue_date} is outside the season ({start} to {end})")
    elif len(years) == 1 and None not in years:
        year = years.pop()
    else:
        year = None
        problems.append("the live files must hold one season (name them by year, or give issue_date)")
    return year, problems


def run(blend, forecasts: dict, *, issue_date=None, resources=None) -> Forecast:
    """Blended probabilities for the live forecasts (model name -> file or files)."""
    from .training import Resources

    resources = resources or Resources.default()
    if isinstance(issue_date, str):
        issue_date = dt.date.fromisoformat(issue_date)
    live = {name: _as_paths(paths) for name, paths in forecasts.items()}
    year, problems = _season_problems(blend.config, issue_date, live)
    problems += blend.check_forecast_files(live)
    if issue_date is not None and not problems:
        for name, paths in live.items():
            dates = sorted({d for p in paths for d in issue_dates(p, blend.config.model(name))})
            if issue_date not in dates:
                nearby = [d.isoformat() for d in dates if abs((d - issue_date).days) <= 7]
                problems.append(
                    f"{name}: the live files have no {issue_date} issue"
                    + (f" (nearby: {', '.join(nearby)})" if nearby else "")
                )
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
    keep = frozenset(blend.units)
    model_onsets = {
        model.name: onsets.forecast(
            config, model, keep_ids=keep, reference=blend.reference, area=domain.AreaFilter.everything(),
            workers=resources.workers, budget_bytes=resources.budget_bytes,
        )
        for model in config.models
    }
    climatology = climatology_stage.build(
        config, blend.observed, [year], test_years=(year, year), workers=resources.workers
    )
    # No trimming: a forecast must not know when the onset came.
    weekly = features.weekly(config, climatology, blend.observed, model_onsets, trim=False)
    if issue_date is not None:
        weekly = weekly[pd.to_datetime(weekly["time"]).dt.date == issue_date]
    rows = weekly.dropna(subset=_input_columns(blend.model, weekly)).reset_index(drop=True)
    if rows.empty:
        raise IncompatibleForecast(["no unit and issue date has every feature the blend needs"])
    return Forecast(weekly=_probabilities(blend, rows), features=rows)


def _input_columns(model, weekly: pd.DataFrame) -> list[str]:
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
