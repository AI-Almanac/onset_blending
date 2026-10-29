"""The KDE climatology's numerics are the legacy climatology_utils'."""
import numpy as np
import pandas as pd
import pytest

from onset_blending._legacy import climatology_utils as legacy
from onset_blending.core import climatology


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_probabilities_are_the_legacy_kdes(seed):
    rng = np.random.default_rng(seed)
    onset_days = rng.normal(30, 12, size=25).round()
    d0 = np.repeat(np.arange(0, 90, 3), 2)
    windows = np.where(np.arange(len(d0)) % 7 == 0, 30, 45)
    windows[5] = -1  # outside every horizon

    new = climatology._predict_pair(climatology.fit_kde(onset_days), d0, windows, 45)
    old = legacy._predict_pair_for_rows(legacy.fit_kde(onset_days), d0, windows, 45)
    legacy._kde_cdf.cache_clear()

    for a, b in zip(new, old):
        np.testing.assert_array_equal(a, b)


def test_too_few_onsets_give_no_kde():
    assert climatology.fit_kde(np.arange(9.0)) is None
    conditional, unconditional = climatology._predict_pair(None, np.arange(3), np.full(3, 10), 10)
    assert np.isnan(conditional).all() and np.isnan(unconditional).all()


@pytest.mark.parametrize("horizons", [None, [
    {"start_md": "05-01", "end_md": "05-31", "forecast_window": 30},
    {"start_md": "06-01", "end_md": "07-15", "forecast_window": 45},
]])
def test_the_issue_grid_is_the_legacy_plans(horizons):
    window = None if horizons else 45
    new = climatology._issue_grid((2019, 2021), "05-01", "07-31", window, horizons)

    grid = legacy.build_issue_grid(2019, 2021, "05-01", "07-31")
    plan = legacy._build_issue_plan(grid, "05-01", window, horizons)
    old = pd.DataFrame({"time": plan["time"], "year": plan["year"], "d0": plan["d0"], "window": plan["window"]})
    pd.testing.assert_frame_equal(new, old, check_exact=True)
