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
