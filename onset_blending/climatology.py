"""The KDE climatology of onset dates: conditional and unconditional onset
probabilities by issue date and lead day, plus windowed variants."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ._legacy import climatology_utils
from .config import BlendConfig


@dataclass
class Climatology:
    conditional: pd.DataFrame | None     # None once released (training.prepare's keep)
    unconditional: pd.DataFrame | None
    train_years: tuple[int, int]
    test_years: tuple[int, int]
    eligible_units: int
    # Conditional climatologies trained on other year windows.
    windows: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)

    def release(self) -> None:
        """Drop the tables, keeping the year windows."""
        self.conditional = self.unconditional = None
        self.windows = {}


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


def _training_onsets(observed_wide: pd.DataFrame, window: tuple[int, int], min_onset_years: int):
    onsets = climatology_utils.read_gt_onset_from_tbl(observed_wide, onset_col="onset_day")
    onsets = climatology_utils.filter_gt_training(onsets, *window)
    counts = onsets.groupby("id")["onset_day"].size()
    eligible = counts[counts >= int(min_onset_years)].index
    onsets = onsets[onsets["id"].isin(eligible)].copy()
    if onsets.empty:
        raise ValueError(
            "No units have enough observed onset years for a climatology "
            f"(min_onset_years={min_onset_years}, years {window[0]}:{window[1]})"
        )
    return onsets, len(eligible)


def build(
    config: BlendConfig,
    observed_wide: pd.DataFrame,
    forecast_years: list[int],
    *,
    test_years: tuple[int, int] | None = None,
    workers: int = 1,
) -> Climatology:
    """The climatologies for the issue dates of the test years.

    test_years overrides the config's (a live season scores one year)."""
    settings = config.climatology
    observed_years = sorted(int(y) for y in observed_wide["year"].dropna().unique())
    train, test = years(config, observed_years, forecast_years, test_years=test_years)
    issue_grid = climatology_utils.build_issue_grid(test[0], test[1], config.onset.season_start, settings.issue_end)
    horizons = [
        {"start_md": h.start, "end_md": h.end, "forecast_window": h.forecast_window}
        for h in settings.horizons
    ] or None
    window_days = None if horizons else config.climatology_forecast_window()
    workers = settings.workers or workers

    def paired(window):
        onsets, eligible = _training_onsets(observed_wide, window, settings.min_onset_years)
        conditional, unconditional = climatology_utils.paired_climatologies(
            onsets, issue_grid, config.onset.season_start, window_days, horizons,
            settings.cv_by_year, workers=workers,
        )
        return conditional, unconditional, eligible

    conditional, unconditional, eligible = paired(train)
    windows = {window: paired(window)[0] for window in settings.windows}
    return Climatology(
        conditional=conditional,
        unconditional=unconditional,
        train_years=train,
        test_years=test,
        eligible_units=eligible,
        windows=windows,
    )
