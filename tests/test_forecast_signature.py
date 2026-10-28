"""A live forecast must have the structure the blend was trained on."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from onset_blending.blend import _signature_differences, forecast_signature
from onset_blending.config import ForecastModel


def model(**changes) -> ForecastModel:
    settings = {"name": "ens", "files": {"paths": ["x.nc"]}, "dims": {"step": "day"}, "lead_days": {"min": 1, "max": 3}}
    settings.update(changes)
    from onset_blending.config import _build

    problems: list[str] = []
    built = _build(ForecastModel, settings, "model", problems)
    assert not problems, problems
    return built


def write(path, *, members=4, leads=4, hour=0, units="mm", lat=(10.0, 10.25), year=2020):
    times = pd.date_range(f"{year}-05-01 {hour:02d}:00", periods=3, freq="D")
    data = np.zeros((len(times), members, leads, len(lat), 2), dtype="float32")
    ds = xr.Dataset(
        {"tp": (("time", "number", "step", "lat", "lon"), data, {"units": units})},
        coords={"time": times, "number": np.arange(members), "step": np.arange(1, leads + 1),
                "lat": list(lat), "lon": [70.0, 70.25]},
    )
    ds.to_netcdf(path)
    return path


def differences(tmp_path, trained_kwargs, live_kwargs, **model_changes):
    m = model(**model_changes)
    trained = forecast_signature(write(tmp_path / "trained_2020.nc", **trained_kwargs), m)
    live = forecast_signature(write(tmp_path / "live_2026.nc", year=2026, **live_kwargs), m)
    return _signature_differences(trained, live)


def test_a_file_like_the_training_files_matches(tmp_path):
    assert differences(tmp_path, {}, {}) == []


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        ({"members": 3}, "members [0, 1, 2], trained on [0, 1, 2, 3]"),
        ({"leads": 2}, "lead days [1, 2], trained on [1, 2, 3]"),
        ({"hour": 12}, "issue hour (UTC) [12], trained on [0]"),
        ({"units": "m"}, "units 'm', trained on 'mm'"),
        ({"lat": (10.0, 10.5)}, "latitudes"),
    ],
)
def test_each_structural_difference_refuses_the_forecast(tmp_path, live, expected):
    [found] = differences(tmp_path, {}, live)
    assert found.startswith(expected)


def test_members_and_lead_days_the_trained_selection_drops_do_not_count(tmp_path):
    # Trained on members 0-1 (max_member 1) and lead days 1-3 of longer files.
    assert differences(tmp_path, {"members": 4, "leads": 5}, {"members": 6, "leads": 4}, max_member=1) == []
    assert differences(tmp_path, {"members": 4}, {"members": 1}, max_member=1) == [
        "members [0], trained on [0, 1]"
    ]
