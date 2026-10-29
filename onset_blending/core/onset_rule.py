"""The onset rule: the first day of a rainfall series that triggers an onset
(a wet window over the threshold, or reaching it) without a dry spell in its
follow-up (none is checked with follow_days 0).

`find_onsets` searches many series at once, for several earliest start days
each, with the legacy find_onsets_batch's numerics: the trigger and the
dry-spell veto come from prefix sums over each series, computed once and
shared by every start day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import OnsetDefinition


@dataclass(frozen=True)
class OnsetRule:
    window: int                 # trigger window, days
    wet_day_min_mm: float
    trigger_rule: str           # all_days_wet | first_day_wet
    wet_day_comparison: str     # gte | gt
    threshold_comparison: str   # gt | gte: the window total against the threshold
    follow_days: int
    followup_anchor: str        # after_trigger | onset_day
    mode: str                   # consecutive_dry | window_sum
    min_dry_days: int
    dry_day_min_mm: float
    sum_window: int
    sum_min_mm: float

    @classmethod
    def of(cls, definition: OnsetDefinition) -> OnsetRule:
        spell = definition.dry_spell
        return cls(
            window=int(definition.window),
            wet_day_min_mm=float(definition.wet_day_min_mm),
            trigger_rule=definition.trigger_rule,
            wet_day_comparison=definition.wet_day_comparison,
            threshold_comparison=definition.threshold_comparison,
            follow_days=int(definition.follow_days),
            followup_anchor=definition.followup_anchor,
            mode=spell.mode,
            min_dry_days=int(spell.min_dry_days),
            dry_day_min_mm=float(spell.dry_day_min_mm if spell.dry_day_min_mm is not None else definition.wet_day_min_mm),
            sum_window=int(spell.sum_window),
            sum_min_mm=float(spell.sum_min_mm),
        )


def _prefix_sums(values: np.ndarray, dtype=None) -> np.ndarray:
    """Row-wise cumulative sums with a leading zero column."""
    summed = np.cumsum(values, axis=1, dtype=dtype)
    return np.concatenate((np.zeros((len(values), 1), dtype=summed.dtype), summed), axis=1)


def find_onsets(series: np.ndarray, thresholds: np.ndarray, start_days: np.ndarray, rule: OnsetRule, *,
                reject_if_short_followup: bool = False, chunk_size: int = 10_000) -> np.ndarray:
    """Onset days (1-based, NaN for none) of each row of `series` (rows,
    days), for each row of `start_days` (the earliest day an onset may fall
    on, per series; one vector or a matrix of alternatives). A series without
    a threshold has no onset; reject_if_short_followup refuses triggers whose
    follow-up runs past the series."""
    series = np.asarray(series, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    start_days = np.asarray(start_days, dtype=float)
    if series.ndim != 2:
        raise ValueError("series must be a two-dimensional rows-by-days array")
    if start_days.ndim == 1:
        start_days = start_days[np.newaxis, :]
    if len(thresholds) != len(series) or start_days.shape[1] != len(series):
        raise ValueError("thresholds and start_days must match the series rows")

    n_rows, n_days = series.shape
    output = np.full((len(start_days), n_rows), np.nan)
    win = rule.window
    max_candidate = n_days - win + 1
    valid_rows = np.flatnonzero(~np.isnan(thresholds))
    if max_candidate < 1 or len(valid_rows) == 0:
        return output

    candidate_days = np.arange(1, max_candidate + 1)
    if rule.followup_anchor == "onset_day":
        follow_start_offset = 0
        full_end_unclipped = candidate_days + rule.follow_days
    elif rule.followup_anchor == "after_trigger":
        follow_start_offset = win
        full_end_unclipped = candidate_days + win - 1 + rule.follow_days
    else:
        raise NotImplementedError(f"Unsupported followup_anchor: {rule.followup_anchor}")
    full_end = np.minimum(n_days, full_end_unclipped)
    full_followup = full_end_unclipped <= n_days
    if rule.wet_day_comparison == "gt":
        compare_wet = np.greater
    elif rule.wet_day_comparison == "gte":
        compare_wet = np.greater_equal
    else:
        raise NotImplementedError(f"Unsupported wet_day_comparison: {rule.wet_day_comparison}")
    compare_total = {"gt": np.greater, "gte": np.greater_equal}.get(rule.threshold_comparison)
    if compare_total is None:
        raise NotImplementedError(f"Unsupported threshold_comparison: {rule.threshold_comparison}")

    for first in range(0, len(valid_rows), int(chunk_size)):
        rows = valid_rows[first:first + int(chunk_size)]
        rain = series[rows]
        rain_cumsum = _prefix_sums(np.where(np.isnan(rain), 0.0, rain))
        window_sums = rain_cumsum[:, win:] - rain_cumsum[:, :-win]

        if rule.trigger_rule == "first_day_wet":
            wet_ok = compare_wet(rain[:, :max_candidate], rule.wet_day_min_mm)
        elif rule.trigger_rule == "all_days_wet":
            wet_cumsum = _prefix_sums(compare_wet(rain, rule.wet_day_min_mm), dtype=np.int32)
            wet_ok = (wet_cumsum[:, win:] - wet_cumsum[:, :-win]) == win
        else:
            raise NotImplementedError(f"Unsupported trigger_rule: {rule.trigger_rule}")

        eligible = wet_ok & compare_total(window_sums, thresholds[rows, np.newaxis]) & ~np.isnan(rain[:, :max_candidate])
        if reject_if_short_followup:
            eligible &= full_followup[np.newaxis, :]

        if rule.mode == "consecutive_dry":
            dry = ~np.isnan(rain) & (rain < rule.dry_day_min_mm)
            dry_starts = np.zeros_like(dry, dtype=np.int8)
            dry_days = rule.min_dry_days
            if n_days >= dry_days:
                dry_cumsum = _prefix_sums(dry, dtype=np.int32)
                all_dry = (dry_cumsum[:, dry_days:] - dry_cumsum[:, :-dry_days]) == dry_days
                first_dry = all_dry.copy()
                if all_dry.shape[1] > 1:
                    first_dry[:, 1:] &= ~dry[:, :all_dry.shape[1] - 1]
                dry_starts[:, :all_dry.shape[1]] = first_dry
            dry_prefix = _prefix_sums(dry_starts, dtype=np.int32)
            lower = candidate_days - 1 + follow_start_offset
            upper = np.minimum(n_days, full_end)
            has_dry_spell = (dry_prefix[:, upper] - dry_prefix[:, lower]) > 0
        elif rule.mode == "window_sum":
            sum_window = rule.sum_window
            if n_days >= sum_window:
                na_cumsum = _prefix_sums(np.isnan(rain), dtype=np.int32)
                follow_sums = rain_cumsum[:, sum_window:] - rain_cumsum[:, :-sum_window]
                follow_has_na = (na_cumsum[:, sum_window:] - na_cumsum[:, :-sum_window]) > 0
                bad_prefix = _prefix_sums(~follow_has_na & (follow_sums < rule.sum_min_mm), dtype=np.int32)
                max_prefix = bad_prefix.shape[1] - 1
                lower = np.minimum(candidate_days - 1 + follow_start_offset, max_prefix)
                upper = np.minimum(max_prefix, np.maximum(0, full_end - sum_window + 1))
                has_dry_spell = (bad_prefix[:, upper] - bad_prefix[:, lower]) > 0
            else:
                has_dry_spell = np.zeros_like(eligible)
        else:
            raise NotImplementedError(f"Unsupported dry-spell mode: {rule.mode}")

        eligible &= ~has_dry_spell
        for variant, starts in enumerate(start_days):
            minimum_start = np.maximum(1, np.ceil(starts[rows]))
            allowed = eligible & (candidate_days[np.newaxis, :] >= minimum_start[:, np.newaxis])
            has_onset = allowed.any(axis=1)
            output[variant, rows] = np.where(has_onset, candidate_days[allowed.argmax(axis=1)], np.nan)
    return output
