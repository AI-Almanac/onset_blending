"""The whole package on a small synthetic region: train, save, load, forecast."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from onset_blending import BlendConfig
from onset_blending.blend import TrainedBlend
from onset_blending.forecast import IncompatibleForecast
from onset_blending.training import Resources, train

LATS, LONS = [10.0, 10.25], [70.0, 70.25]
OBS_YEARS = range(2000, 2014)
FORECAST_YEARS = range(2010, 2014)
LEAD_DAYS = 40


def onset_date(year: int, cell: int, shift: int = 0) -> dt.date:
    """Onsets between late May and early July, varying by year and cell."""
    return dt.date(year, 5, 25) + dt.timedelta(days=(year * 7 + cell * 5) % 40 + shift)


def rain(dates, onset: dt.date) -> np.ndarray:
    """Dry before the onset, 10 mm a day for its first 3 days, then 6 mm a day."""
    days = np.array([(d - onset).days for d in dates])
    return np.where(days < 0, 0.0, np.where(days < 3, 10.0, 6.0)).astype("float32")


def write_observations(folder):
    folder.mkdir()
    for year in OBS_YEARS:
        dates = [dt.date(year, 4, 1) + dt.timedelta(days=i) for i in range(200)]
        data = np.stack([
            np.stack([rain(dates, onset_date(year, i * 2 + j)) for j in range(len(LONS))], axis=-1)
            for i in range(len(LATS))
        ], axis=1)
        xr.Dataset(
            {"precip": (("time", "lat", "lon"), data)},
            coords={"time": pd.to_datetime(dates), "lat": LATS, "lon": LONS},
        ).to_netcdf(folder / f"{year}.nc")


def write_forecasts(folder, years=FORECAST_YEARS, members=3):
    folder.mkdir(exist_ok=True)
    for year in years:
        issues = [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), members, LEAD_DAYS, len(LATS), len(LONS)), dtype="float32")
        for t, issue in enumerate(issues):
            leads = [issue + dt.timedelta(days=d) for d in range(1, LEAD_DAYS + 1)]
            for m in range(members):
                for i in range(len(LATS)):
                    for j in range(len(LONS)):
                        data[t, m, :, i, j] = rain(leads, onset_date(year, i * 2 + j, shift=3 * (m - 1)))
        xr.Dataset(
            {"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
            coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                    "day": np.arange(1, LEAD_DAYS + 1), "lat": LATS, "lon": LONS},
        ).to_netcdf(folder / f"{year}.nc")


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("region")
    write_observations(root / "obs")
    write_forecasts(root / "ens")
    config = BlendConfig.from_dict({
        "name": "synthetic",
        "data_root": str(root),
        "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}},
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                    "rain_predictors": [{"agg": "diff"}]}],
        "onset": {"threshold": {"mm": 20},
                  "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
        "climatology": {"forecast_window": LEAD_DAYS, "min_onset_years": 5, "cv_by_year": False},
        "evaluation": {
            "years": {"training": "2010:2012", "cv_holdout": "2010:2012", "true_holdout": 2013},
            "formulas": {"blended_model": "outcome ~ prob_clim_qx + diff_ens_qx"},
        },
    })
    result = train(config, resources=Resources(workers=1, budget_bytes=None))
    return root, config, result


def test_training_scores_the_blend_against_climatology(region):
    _, _, result = region
    pooled = result.evaluation.pooled.set_index("model")
    assert {"unc_clim_raw", "clim_raw", "blended_model"} <= set(pooled.index)
    assert result.evaluation.yearly["year"].nunique() == 4


def test_a_saved_blend_forecasts_like_its_model(region, tmp_path):
    root, _, result = region
    out = result.save(tmp_path / "trained")
    blend = TrainedBlend.load(out / "trained.blend")

    forecast = blend.forecast({"ens": root / "ens" / "2013.nc"})

    probabilities = forecast.weekly.filter(like="cv_")
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    direct = result.blend.model.predict(forecast.features)
    assert np.array_equal(direct[probabilities.columns].to_numpy(), probabilities.to_numpy())
    one_date = blend.forecast({"ens": root / "ens" / "2013.nc"}, issue_date="2013-05-15")
    assert set(pd.to_datetime(one_date.weekly["time"]).dt.date) == {dt.date(2013, 5, 15)}


def test_a_forecast_with_other_members_is_refused(region, tmp_path):
    root, _, result = region
    write_forecasts(tmp_path / "live", years=[2014], members=2)

    with pytest.raises(IncompatibleForecast) as refused:
        result.blend.forecast({"ens": tmp_path / "live" / "2014.nc"})

    assert any("members [0, 1], trained on [0, 1, 2]" in p for p in refused.value.problems)


def test_a_damaged_blend_is_refused(region, tmp_path):
    _, _, result = region
    path = result.blend.save(tmp_path / "trained.blend")
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(Exception):
        TrainedBlend.load(path)
