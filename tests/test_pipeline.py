"""The whole package on a small synthetic region: train, save, load, forecast."""
import dataclasses
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from onset_blending import BlendConfig, onsets, readers
from onset_blending.blend import TrainedBlend
from onset_blending.config import FileSet
from onset_blending import forecast as forecast_module
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


def write_forecasts(folder, years=FORECAST_YEARS, members=3, issues_of=None, name=None,
                    lead_days=LEAD_DAYS, lats=LATS, lons=LONS):
    """{year}.nc season files; issues_of(year) and name(year) give other issues and file names."""
    folder.mkdir(exist_ok=True)
    for year in years:
        issues = issues_of(year) if issues_of else [dt.date(year, 5, 1) + dt.timedelta(days=7 * k) for k in range(13)]
        data = np.zeros((len(issues), members, lead_days, len(lats), len(lons)), dtype="float32")
        for t, issue in enumerate(issues):
            day = dt.date(issue.year, issue.month, issue.day)  # issues may carry a time of day
            leads = [day + dt.timedelta(days=d) for d in range(1, lead_days + 1)]
            for m in range(members):
                for i in range(len(lats)):
                    for j in range(len(lons)):
                        data[t, m, :, i, j] = rain(leads, onset_date(year, i * 2 + j, shift=3 * (m - 1)))
        xr.Dataset(
            {"tp": (("time", "number", "day", "lat", "lon"), data, {"units": "mm"})},
            coords={"time": pd.to_datetime(issues), "number": np.arange(members),
                    "day": np.arange(1, lead_days + 1), "lat": lats, "lon": lons},
        ).to_netcdf(folder / (name(year) if name else f"{year}.nc"))


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


def test_one_initialisation_per_model_is_forecast_for_its_issue_date(region, tmp_path):
    _, _, result = region
    init = dt.date(2014, 6, 9)
    write_forecasts(tmp_path, years=[2014], issues_of=lambda _: [init], name=lambda _: "latest.nc")

    forecast = result.blend.forecast({"ens": tmp_path / "latest.nc"})

    assert set(pd.to_datetime(forecast.weekly["time"]).dt.date) == {init}
    assert len(forecast.weekly) == len(LATS) * len(LONS)


def test_a_live_file_holding_more_forecasts_what_was_trained(region, tmp_path):
    # More members, longer leads and a larger domain than training: trimmed.
    _, _, result = region
    init = dict(years=[2014], issues_of=lambda _: [dt.date(2014, 6, 9)])
    write_forecasts(tmp_path / "trained", **init)
    write_forecasts(tmp_path / "bigger", members=5, lead_days=LEAD_DAYS + 6,
                    lats=LATS + [10.5], lons=LONS + [70.5], **init)

    trained = result.blend.forecast({"ens": tmp_path / "trained" / "2014.nc"})
    bigger = result.blend.forecast({"ens": tmp_path / "bigger" / "2014.nc"})

    pd.testing.assert_frame_equal(bigger.weekly, trained.weekly, check_exact=True)


def test_the_order_of_a_files_dimensions_does_not_matter(region, tmp_path):
    # Lead days, latitudes and issues reversed: the same forecast.
    _, _, result = region
    write_forecasts(tmp_path / "sorted", years=[2014])
    with xr.open_dataset(tmp_path / "sorted" / "2014.nc") as ds:
        reversed_ = ds.isel(day=slice(None, None, -1), lat=slice(None, None, -1), time=slice(None, None, -1)).load()
    (tmp_path / "reversed").mkdir()
    reversed_.to_netcdf(tmp_path / "reversed" / "2014.nc")

    expected = result.blend.forecast({"ens": tmp_path / "sorted" / "2014.nc"})
    got = result.blend.forecast({"ens": tmp_path / "reversed" / "2014.nc"})

    pd.testing.assert_frame_equal(got.weekly, expected.weekly, check_exact=True)


def test_observation_dates_that_mark_the_day_itself_move_to_the_forecasts_labels(region, tmp_path):
    # A calendar-day product (date = the day the rain fell) labels each day a
    # day earlier than forecasts do (the UTC day ending at issue + d).
    _, config, _ = region
    start = dataclasses.replace(config, observations=dataclasses.replace(config.observations, date_marks="start"))

    marked_end = onsets.observed(config).frame().set_index(["id", "year"])
    marked_start = onsets.observed(start).frame().set_index(["id", "year"])

    shifted = pd.to_datetime(marked_start["onset_date"]) - pd.to_datetime(marked_end["onset_date"])
    assert (shifted == pd.Timedelta(days=1)).all()
    assert (marked_start["onset_day"] - marked_end["onset_day"] == 1).all()


def test_shifted_observation_dates_keep_their_season(tmp_path):
    times = pd.date_range("2010-12-30", periods=3, freq="D")
    xr.Dataset({"precip": (("time", "lat", "lon"), np.zeros((3, 1, 1), dtype="float32"))},
               coords={"time": times, "lat": [10.0], "lon": [70.0]}).to_netcdf(tmp_path / "obs.nc")

    read = readers.read_observations(tmp_path / "obs.nc", "precip", {}, shift_days=1)

    assert pd.DatetimeIndex(read.dates).strftime("%m-%d").tolist() == ["12-31", "01-01", "01-02"]
    assert read.years.tolist() == [2010, 2010, 2011]


def test_lead_days_must_be_consecutive_and_distinct():
    positions, days = readers.lead_days_in_order(np.arange(3), [3, 1, 2], "f.nc")
    assert days == [1, 2, 3] and positions.tolist() == [1, 2, 0]
    for bad in ([1, 2, 4], [1, 1, 2], [1, None, 2]):
        with pytest.raises(ValueError, match="f.nc: lead days"):
            readers.lead_days_in_order(np.arange(3), bad, "f.nc")


def test_two_issues_on_one_date_are_refused(region, tmp_path):
    root, config, _ = region
    write_forecasts(tmp_path, years=[2014], issues_of=lambda _: [dt.datetime(2014, 6, 9), dt.datetime(2014, 6, 9, 12)])
    model = dataclasses.replace(config.models[0], files=FileSet(paths=(str(tmp_path / "2014.nc"),)))

    with pytest.raises(ValueError, match="several issues on the same date \\(2014-06-09\\)"):
        onsets.forecast(config, model)


def test_models_initialised_on_different_dates_are_refused(region):
    _, config, _ = region
    held = {"ens": [dt.date(2014, 6, 9)], "det": [dt.date(2014, 6, 2)]}

    issue_date, _, problems = forecast_module._issue_problems(config, None, held)

    assert issue_date is None
    assert problems == [
        "the live files are initialised on different dates (ens 2014-06-09, det 2014-06-02); "
        "every model needs the same one"
    ]


def test_an_initialisation_outside_the_season_is_refused(region):
    _, config, _ = region

    _, _, problems = forecast_module._issue_problems(config, None, {"ens": [dt.date(2014, 11, 3)]})

    assert len(problems) == 1 and problems[0].startswith("issue date 2014-11-03 is outside the season")


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


def test_stage_tables_are_kept_only_when_asked(region):
    _, config, result = region
    from onset_blending.training import STAGE_TABLES, prepare

    assert all(f.data is None for f in result.inputs.forecasts.values())  # train releases them
    kept = prepare(config, resources=Resources(workers=1, budget_bytes=None), keep=STAGE_TABLES)
    assert all(f.data is not None for f in kept.forecasts.values())
    assert kept.climatology.data is not None


def test_the_day_level_table_is_built_on_request(region):
    _, config, result = region
    from onset_blending.training import prepare

    assert result.inputs.combined is None
    with_daily = prepare(config, resources=Resources(workers=1, budget_bytes=None), daily=True)
    combined = with_daily.combined
    assert len(combined) == len(with_daily.weekly)
    assert {"clim_p_onset_day_1", "ens_p_onset_day_41plus", "ens_rain_sd_day_1", "true_onset_date"} <= set(combined)
