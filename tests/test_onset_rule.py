"""The onset rule's batched search is the legacy find_onsets_batch."""
import numpy as np
import pytest

from onset_blending import config as cfg
from _legacy import onset_utils
from onset_blending.core.onset_rule import OnsetRule, find_onsets

DEFINITIONS = [
    cfg.OnsetDefinition(),
    cfg.OnsetDefinition(window=3, follow_days=21, dry_spell=cfg.DrySpell(min_dry_days=5, dry_day_min_mm=1.0)),
    cfg.OnsetDefinition(trigger_rule="first_day_wet", wet_day_comparison="gt", followup_anchor="onset_day",
                        dry_spell=cfg.DrySpell(mode="window_sum", sum_window=10, sum_min_mm=5.0)),
]


def legacy_params(definition):
    spell = definition.dry_spell
    dry_spell = {"mode": spell.mode, "min_dry_days": spell.min_dry_days, "sum_window": spell.sum_window,
                 "sum_min_mm": spell.sum_min_mm}
    if spell.dry_day_min_mm is not None:
        dry_spell["dry_day_min_mm"] = spell.dry_day_min_mm
    return onset_utils.read_onset_params({"options": {"window": definition.window, "onset_definition": {
        "wet_day_min_mm": definition.wet_day_min_mm, "trigger_rule": definition.trigger_rule,
        "wet_day_comparison": definition.wet_day_comparison, "follow_days": definition.follow_days,
        "followup_anchor": definition.followup_anchor, "dry_spell": dry_spell}}})


@pytest.mark.parametrize("definition", DEFINITIONS)
@pytest.mark.parametrize("short", [False, True])
def test_onsets_are_the_legacy_searchs(definition, short):
    rng = np.random.default_rng(0)
    series = rng.gamma(0.6, 6.0, size=(300, 60))
    series[rng.random(series.shape) < 0.03] = np.nan
    thresholds = rng.uniform(10, 30, size=300)
    thresholds[:5] = np.nan
    starts = np.vstack([np.ones(300), rng.integers(1, 30, 300), np.full(300, 9999.0)])

    new = find_onsets(series, thresholds, starts, OnsetRule.of(definition), reject_if_short_followup=short, chunk_size=64)
    old = onset_utils.find_onsets_batch(series, thresholds, starts, legacy_params(definition),
                                        reject_if_short_followup=short, chunk_size=64)
    np.testing.assert_array_equal(new, old)


def test_a_total_that_reaches_the_threshold_triggers_only_with_gte():
    series = np.array([[0, 0, 5, 5, 5, 5, 5, 0, 0, 0]], dtype=float)  # a 5-day total of exactly 25
    starts = np.ones(1)
    reaches = cfg.OnsetDefinition(window=5, trigger_rule="first_day_wet", threshold_comparison="gte", follow_days=0)
    exceeds = cfg.OnsetDefinition(window=5, trigger_rule="first_day_wet", follow_days=0)

    assert find_onsets(series, np.array([25.0]), starts, OnsetRule.of(reaches))[0, 0] == 3
    assert np.isnan(find_onsets(series, np.array([25.0]), starts, OnsetRule.of(exceeds))[0, 0])


ICPAC_WITH_DRY_SPELL = cfg.OnsetDefinition(
    window=3, trigger_rule="first_day_wet", follow_days=21, followup_anchor="onset_day",
    dry_spell=cfg.DrySpell(min_dry_days=7, dry_day_min_mm=1.0, contained=True),
)


@pytest.mark.parametrize(("dry_from", "vetoed"), [(6, True), (15, True), (16, False)])
def test_a_contained_dry_run_must_lie_within_the_days_after_the_onset(dry_from, vetoed):
    # A 25 mm onset on the first day, then wet but for 7 dry days starting
    # `dry_from` days after it: within the 21 days after the onset day only
    # when they end by day 21 after it.
    series = np.full((1, 60), 2.0)
    series[0, :3] = [10.0, 10.0, 5.0]
    series[0, dry_from:dry_from + 7] = 0.0
    found = find_onsets(series, np.array([20.0]), np.ones(1), OnsetRule.of(ICPAC_WITH_DRY_SPELL))[0, 0]
    assert bool(found != 1) is vetoed


def test_a_contained_dry_run_matches_a_direct_check():
    rng = np.random.default_rng(3)
    series = np.where(rng.random((400, 70)) < 0.45, 0.0, rng.gamma(0.8, 8.0, size=(400, 70)))
    found = find_onsets(series, np.full(400, 20.0), np.ones(400), OnsetRule.of(ICPAC_WITH_DRY_SPELL))[0]

    for row, onset in zip(series, found):
        expected = np.nan
        for d in range(len(row) - 2):  # 0-based candidate
            if row[d] < 1.0 or row[d:d + 3].sum() <= 20.0:
                continue
            after = row[d + 1:d + 22] < 1.0
            runs = np.convolve(after, np.ones(7, dtype=int), "valid") == 7 if len(after) >= 7 else []
            if not np.any(runs):
                expected = d + 1
                break
        assert (np.isnan(onset) and np.isnan(expected)) or onset == expected


ICPAC = cfg.OnsetDefinition(window=5, trigger_rule="first_day_wet", min_wet_days=3, threshold_comparison="gte",
                            follow_days=0)


@pytest.mark.parametrize(("series", "onset"), [
    ([8, 0, 6, 0, 6, 0, 0], 1),        # 20 mm over 5 days, 3 rain days, from a rain day
    ([8, 0, 6, 0, 5.9, 0, 0], np.nan),  # 19.9 mm
    ([18, 0, 0, 0, 2, 0, 0], np.nan),   # 2 rain days
    ([0, 8, 6, 6, 0, 0, 0], 2),        # not from the dry first day: from the next, a rain day
])
def test_icpac_needs_20_mm_over_5_days_with_3_rain_days_from_a_rain_day(series, onset):
    found = find_onsets(np.array([series], dtype=float), np.array([20.0]), np.ones(1), OnsetRule.of(ICPAC))[0, 0]
    assert (np.isnan(found) and np.isnan(onset)) or found == onset


def test_min_wet_days_matches_a_direct_check():
    rng = np.random.default_rng(4)
    series = np.where(rng.random((500, 60)) < 0.5, 0.0, rng.gamma(0.7, 7.0, size=(500, 60)))
    found = find_onsets(series, np.full(500, 20.0), np.ones(500), OnsetRule.of(ICPAC))[0]
    for row, onset in zip(series, found):
        expected = next(
            (d + 1 for d in range(len(row) - 4)
             if row[d] >= 1 and row[d:d + 5].sum() >= 20 and (row[d:d + 5] >= 1).sum() >= 3),
            np.nan,
        )
        assert (np.isnan(onset) and np.isnan(expected)) or onset == expected
