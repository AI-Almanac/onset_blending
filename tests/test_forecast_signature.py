"""A live forecast must cover what the blend was trained on: the same units,
aggregation, resolution and issue hour, and at least the trained lead days,
members and domain (more is trimmed)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from onset_blending.trained import _signature_differences, forecast_signature
from onset_blending.config import ForecastModel


def model(**changes) -> ForecastModel:
    settings = {"name": "ens", "files": {"paths": ["x.nc"]}, "dims": {"step": "day"}, "lead_days": {"min": 1, "max": 3}}
    settings.update(changes)
    from onset_blending.config import _build

    problems: list[str] = []
    built = _build(ForecastModel, settings, "model", problems)
    assert not problems, problems
    return built


WINDOW = "End of the UTC calendar-day aggregation window (00:00 UTC of the following date)"


def write(path, *, members=4, leads=4, hour=0, units="mm", lat=(10.0, 10.25), lon=(70.0, 70.25), year=2020,
          lead_dim="step", method="sum", window=WINDOW):
    times = pd.date_range(f"{year}-05-01 {hour:02d}:00", periods=3, freq="D")
    dims = ("time",) + (("number",) if members else ()) + (lead_dim, "lat", "lon")
    shape = (len(times),) + ((members,) if members else ()) + (leads, len(lat), len(lon))
    attrs = {"units": units} if units else {}
    if method:
        attrs["cell_methods"] = f"{lead_dim}: {method}"
    coords = {"time": times, lead_dim: (lead_dim, np.arange(1, leads + 1), {"description": window} if window else {}),
              "lat": list(lat), "lon": list(lon)}
    if members:
        coords["number"] = np.arange(members)
    xr.Dataset({"tp": (dims, np.zeros(shape, dtype="float32"), attrs)}, coords=coords).to_netcdf(path)
    return path


def differences(tmp_path, trained_kwargs, live_kwargs, **model_changes):
    """As check_forecast_files compares them: the trained signature after the
    model's selection, the live one as the file holds it."""
    m = model(**model_changes)
    trained = forecast_signature(write(tmp_path / "trained_2020.nc", **trained_kwargs), m)
    live = forecast_signature(write(tmp_path / "live_2026.nc", year=2026, **live_kwargs), m, selected=False)
    return _signature_differences(trained, live)


def test_a_file_like_the_training_files_matches(tmp_path):
    assert differences(tmp_path, {}, {}) == []


@pytest.mark.parametrize(
    ("live", "expected"),
    [
        ({"members": 3}, "members [0, 1, 2], trained on [0, 1, 2, 3]"),
        ({"leads": 2}, "lead days [1, 2], trained on [1, 2, 3]"),
        ({"hour": 12}, "issue hour (UTC) [12], trained on [0]"),
        ({"units": None}, "rainfall units 'undeclared (x1)', trained on 'mm'"),
        ({"lat": (10.0, 10.5)}, "latitudes every 0.5°, trained on every 0.25° (another resolution)"),
        ({"lat": (10.0,)}, "latitudes 10–10, trained on 10–10.25: the trained domain is not covered"),
        ({"members": 0}, "members None, trained on [0, 1, 2, 3]"),
        ({"method": "mean"}, "daily aggregation"),
        ({"window": "Accumulation over the 24 h after the lead"}, "daily aggregation"),
        ({"window": None, "method": None}, "daily aggregation"),
    ],
)
def test_each_structural_difference_refuses_the_forecast(tmp_path, live, expected):
    [found] = differences(tmp_path, {}, live)
    assert found.startswith(expected)


def test_members_and_lead_days_the_trained_selection_drops_do_not_count(tmp_path):
    # Trained on members 0-1 (max_member 1) and lead days 1-3 of longer files.
    assert differences(tmp_path, {"members": 4, "leads": 5}, {"members": 6, "leads": 4}, max_member=1) == []
    assert differences(tmp_path, {"members": 4}, {"members": 1}, max_member=1) == [
        "members [0], trained on [0, 1] (missing [1])"
    ]


def test_a_live_file_holding_more_is_covered(tmp_path):
    # More members, longer leads and a larger domain at the same resolution.
    bigger = {"members": 8, "leads": 10, "lat": (9.75, 10.0, 10.25, 10.5), "lon": (69.75, 70.0, 70.25)}
    assert differences(tmp_path, {}, bigger) == []


def test_a_smaller_domain_is_refused(tmp_path):
    trained = {"lat": (10.0, 10.25, 10.5)}
    assert differences(tmp_path, trained, {"lat": (10.0, 10.25)}) == [
        "latitudes 10–10.25, trained on 10–10.5: the trained domain is not covered"
    ]


def test_files_in_other_formats_read_alike(tmp_path):
    # An archive in metres with its own lead-day name; a live file in mm on `day`.
    m = model(dims={})
    archive = write(tmp_path / "archive_2020.nc", units="m", lead_dim="prediction_timedelta_daily")
    live = write(tmp_path / "live_2026.nc", year=2026, units="mm", lead_dim="day")
    assert _signature_differences(forecast_signature(archive, m), forecast_signature(live, m, selected=False)) == []
    assert forecast_signature(archive, m)["units"] == "mm"
