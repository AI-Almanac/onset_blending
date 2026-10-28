"""Features: the day-level table joining climatologies, forecasts and observed
onsets (today's combine), and the weekly table the blend trains on (connect)."""

from __future__ import annotations

from functools import reduce

import pandas as pd

from . import _parallel, _specs
from ._legacy import combine_forecasts_utils as combine_utils
from ._legacy.connect_utils import weekly_from_daylevel
from .climatology import Climatology
from .config import BlendConfig
from .onsets import ForecastOnsets

_KEYS = ["id", "time", "year"]
# Row position in the conditional climatology, carried through a year's
# joins so the per-year combine reproduces a one-shot combine's row order.
_CLIM_ROW = "__clim_row__"
# Peak memory of one year's combine relative to its inputs (measured up to
# 2.9x on India with two models), and the weekly table's row-chunk size.
_COMBINE_BYTES_PER_INPUT_BYTE = 4
_WEEKLY_CHUNK_BYTES = 2 * 2**30


def _family(model: str, max_day: int, has_spread: bool) -> dict:
    """The combine's columns for one model (sources are added per year)."""
    daily = [
        {"col": "predicted_prob", "out": "p_onset", "add_plus": True},
        {"col": "predicted_prob_fixed_cutoff", "out": "p_onset_fixed_cutoff", "add_plus": True},
        {"col": "predicted_prob_ref", "out": "p_onset_ref", "add_plus": True},
        {"col": "forecast_rain", "out": "rain_mean", "add_plus": False},
        {"col": "frac_raining", "out": "frac_raining", "add_plus": False},
    ]
    if has_spread:
        daily.append({"col": "forecast_rain_sd", "out": "rain_sd", "add_plus": False})
    return {
        "max_day": int(max_day),
        "constants": [
            {"col": "onset_thresh", "out": "onset_thresh"},
            {"col": "ref_onset_date", "out": "ref_onset_date"},
        ],
        "daily": daily,
    }


def _combine_year(task: dict, context: dict) -> pd.DataFrame:
    """The day-level rows for one partition of years."""
    years = task["years"]
    span = f"{years[0]}:{years[-1]}"
    clim, *others = [
        combine_utils.read_and_format_climatology_wide(table, out_prefix=f"{key}_p_onset")
        for key, table in task["climatologies"]
    ]
    clim[_CLIM_ROW] = clim.index
    families = {
        model: combine_utils.format_forecast_family(
            model,
            {**context["families"][model], "sources": [{"frame": part, "years": span} for part in parts]},
        )
        for model, parts in task["forecasts"].items()
    }
    daily = reduce(
        lambda left, right: left.merge(right, on=_KEYS, how=context["join"]),
        [clim, *others, *(family["daily"] for family in families.values())],
    )
    daily["year"] = daily["year"].astype(int)
    combined = daily.merge(context["truth"], on=["id", "year"], how="left")
    if context["trim"]:
        keep = combined["true_onset_date"].isna() | (
            pd.to_datetime(combined["time"]) <= pd.to_datetime(combined["true_onset_date"])
        )
        combined = combined.loc[keep].copy()
    constants = [family["constants"] for family in families.values()]
    if constants:
        merged = reduce(lambda left, right: left.merge(right, on=_KEYS, how="outer"), constants)
        combined = combined.merge(merged, on=_KEYS, how="left")
    return combined


def combine(
    config: BlendConfig,
    climatology: Climatology,
    observed_wide: pd.DataFrame,
    forecasts: dict[str, ForecastOnsets],
    *,
    workers: int = 1,
    budget_bytes: int | None = None,
) -> pd.DataFrame:
    """The day-level table: every climatology joined to each model's daily
    probabilities and rainfall, then the observed onset and model constants."""
    join = "outer" if config.features.join == "full" else "inner"
    for name, onsets in forecasts.items():
        if not onsets.years:
            raise ValueError(f"Model {name!r} has no forecast years")
    all_years = sorted(set().union(*(set(onsets.years) for onsets in forecasts.values())))
    # An inner join keeps the climatology's row order and splits by year, so
    # it runs a year at a time; an outer join reorders keys, so it runs once.
    partitions = [[year] for year in all_years] if join == "inner" else [all_years]

    climatologies = [
        (_specs.CLIMATOLOGY, climatology.conditional.reset_index(drop=True)),
        (_specs.UNCONDITIONAL_CLIMATOLOGY, climatology.unconditional.reset_index(drop=True)),
        *((_specs.climatology_key(window), table.reset_index(drop=True)) for window, table in climatology.windows.items()),
    ]
    tasks = []
    for years in partitions:
        parts = {
            name: [p for p in onsets.parts if set(p["year"].dropna().astype(int).unique()) & set(years)]
            for name, onsets in forecasts.items()
        }
        # The inner join keeps no row for a year some model lacks.
        if join == "inner" and not all(parts.values()):
            continue
        tasks.append({
            "years": years,
            "climatologies": [(key, table[table["year"].isin(years)]) for key, table in climatologies],
            "forecasts": parts,
        })
    context = {
        "families": {
            name: _family(name, config.model(name).lead_days.max, onsets.has_spread)
            for name, onsets in forecasts.items()
        },
        "truth": combine_utils.read_ground_truth_wide(observed_wide),
        "join": join,
        "trim": config.features.trim_after_true_onset,
    }
    count = _parallel.workers_within([_task_bytes(task) for task in tasks], workers, budget_bytes)
    combined = pd.concat(_parallel.map_ordered(_combine_year, tasks, context, count), ignore_index=True)
    if join == "inner":
        combined = combined.sort_values(_CLIM_ROW, kind="stable", ignore_index=True)
    return combined.drop(columns=_CLIM_ROW)


def _task_bytes(task: dict) -> int:
    frames = [table for _, table in task["climatologies"]]
    frames += [part for parts in task["forecasts"].values() for part in parts]
    return _COMBINE_BYTES_PER_INPUT_BYTE * sum(int(f.memory_usage(deep=True).sum()) for f in frames)


def weekly(config: BlendConfig, combined: pd.DataFrame) -> pd.DataFrame:
    """The weekly table: outcomes, climatology logits, binned forecast
    probabilities and rain predictors per unit and issue date.

    Each weekly row derives from its own day-level row, so the table is built a
    chunk of rows at a time, which bounds the working memory."""
    spec = _specs.connect_spec(config)
    row_bytes = max(int(combined.memory_usage(deep=True).sum()) // max(len(combined), 1), 1)
    rows = max(_WEEKLY_CHUNK_BYTES // row_bytes, 1)
    parts = [
        weekly_from_daylevel(combined.iloc[start : start + rows], spec)
        for start in range(0, max(len(combined), 1), rows)
    ]
    table = pd.concat(parts, ignore_index=True)
    table.attrs.update(parts[0].attrs)
    return table
