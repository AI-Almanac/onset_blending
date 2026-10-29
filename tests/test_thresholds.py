"""Onset thresholds computed from the observations (onset.threshold.climatology)."""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from onset_blending import BlendConfig, ConfigError, TrainedBlend
from onset_blending.core.reference import Reference
from onset_blending.training import Resources, train

from test_pipeline import LEAD_DAYS, write_forecasts, write_observations

MORON_ROBERTSON_MOK = {
    "window": 5, "trigger_rule": "first_day_wet", "wet_day_comparison": "gte",
    "threshold_comparison": "gte", "follow_days": 0,
}


def config_data(root, threshold):
    return {
        "name": "mr",
        "data_root": str(root),
        "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}},
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                    "rain_predictors": [{"agg": "diff"}]}],
        "onset": {"threshold": threshold, "reference_onset": {"month_day": "06-02"}, "fixed_cutoff": "06-02",
                  "definition": MORON_ROBERTSON_MOK},
        "climatology": {"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": False},
        "evaluation": {"years": {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013},
                       "formulas": {"blended_model": "outcome ~ prob_clim_qx + diff_ens_qx"}},
    }


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("mr")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    return root


def test_each_cells_threshold_is_its_mean_rain_over_the_window(region):
    window = {"days": 5, "start": "05-01", "end": "09-30", "years": "2001:2008"}
    config = BlendConfig.from_dict(config_data(region, {"climatology": window}))

    thresholds = Reference.resolve(config).thresholds

    totals, days = {}, {}
    for year in range(2001, 2009):
        with xr.open_dataset(region / "obs" / f"{year}.nc") as ds:
            rain = ds["precip"].sel(time=slice(f"{year}-05-01", f"{year}-09-30"))
            for lat in ds["lat"].values:
                for lon in ds["lon"].values:
                    key = f"{lat:.2f}_{lon:.2f}"
                    cell = rain.sel(lat=lat, lon=lon).values.astype(float)
                    totals[key] = totals.get(key, 0.0) + np.nansum(cell)
                    days[key] = days.get(key, 0) + int(np.isfinite(cell).sum())
    expected = pd.DataFrame({"id": sorted(totals), "onset_thresh": [5 * totals[k] / days[k] for k in sorted(totals)]})
    pd.testing.assert_frame_equal(thresholds.reset_index(drop=True), expected)


def test_scaled_climatological_thresholds_are_clipped(region):
    window = {"start": "05-01", "end": "09-30", "years": [2001, 2008]}
    config = BlendConfig.from_dict(config_data(region, {"climatology": window, "scale": {"factor": 0.5, "max": 10}}))

    thresholds = Reference.resolve(config).thresholds["onset_thresh"]

    assert (thresholds <= 10).all()


def test_a_threshold_window_without_observations_is_refused(region):
    config = BlendConfig.from_dict(config_data(region, {"climatology": {"years": "1950:1960"}}))
    with pytest.raises(ValueError, match="no observations"):
        Reference.resolve(config)


def test_a_threshold_is_one_of_a_value_a_table_or_a_climatology(region):
    with pytest.raises(ConfigError, match="exactly one of 'mm', 'file' or 'climatology'"):
        BlendConfig.from_dict(config_data(region, {"mm": 20, "climatology": {"years": "2001:2008"}}))


def test_a_blend_keeps_the_thresholds_it_computed(region, tmp_path):
    window = {"start": "04-01", "end": "10-31", "years": "2001:2009"}
    config = BlendConfig.from_dict(config_data(region, {"climatology": window}))

    result = train(config, resources=Resources(workers=1, budget_bytes=None))
    blend = TrainedBlend.load(result.save(tmp_path / "trained") / "trained.blend")

    pd.testing.assert_frame_equal(blend.reference.thresholds, result.inputs.reference.thresholds)
    onsets = result.inputs.observed.onset_dates()
    found = onsets[~np.isnat(onsets)].astype("datetime64[D]")
    assert len(found) and all(pd.Timestamp(d).strftime("%m-%d") >= "06-02" for d in found)
    forecast = blend.forecast({"ens": region / "ens" / "2013.nc"})
    assert len(forecast.weekly) and forecast.weekly.filter(like="cv_").notna().all().all()


def test_computed_thresholds_are_kept_and_reused(region, tmp_path, monkeypatch):
    from onset_blending.core import onsets

    window = {"start": "05-01", "end": "09-30", "years": "2001:2008"}
    config = BlendConfig.from_dict(config_data(region, {"climatology": window}))
    first = Reference.resolve(config, cache_dir=tmp_path).thresholds
    assert len(list((tmp_path / "onset-thresholds").glob("*.nc"))) == 1

    def refuse(*args, **kwargs):
        raise AssertionError("the thresholds were recomputed")

    monkeypatch.setattr(onsets, "_compute_thresholds", refuse)
    again = Reference.resolve(config, cache_dir=tmp_path).thresholds
    pd.testing.assert_frame_equal(again, first, check_exact=True)

    # Other inputs, other thresholds: another averaging window, or a changed file.
    other = BlendConfig.from_dict(config_data(region, {"climatology": {**window, "end": "08-31"}}))
    with pytest.raises(AssertionError, match="recomputed"):
        Reference.resolve(other, cache_dir=tmp_path)
    path = region / "obs" / "2005.nc"
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    with pytest.raises(AssertionError, match="recomputed"):
        Reference.resolve(config, cache_dir=tmp_path)


def test_training_keeps_its_thresholds_in_the_resources_cache(region, tmp_path):
    window = {"start": "04-01", "end": "10-31", "years": "2001:2009"}
    config = BlendConfig.from_dict(config_data(region, {"climatology": window}))

    train(config, resources=Resources(workers=1, budget_bytes=None, cache_dir=tmp_path))

    assert len(list((tmp_path / "onset-thresholds").glob("*.nc"))) == 1
