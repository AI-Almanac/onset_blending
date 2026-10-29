"""The NetCDF primitives are the legacy nc_utils'."""
import numpy as np
import pandas as pd
import pytest

from onset_blending._legacy import nc_utils
from onset_blending.core import io


@pytest.mark.parametrize("units", ["days since 2000-01-01", "hours since 1990-06-01 06:00:00", "seconds since 2020-01-01T00:00"])
def test_times_are_the_legacy_dates(units):
    values = np.array([0, 1.5, 36, 400])
    pd.testing.assert_index_equal(io.time_to_dates(values, units), nc_utils.nc_time_to_dates(values, units))


def test_grids_and_days_are_the_legacy_readers():
    values = [np.array([10.0, 10.5]), np.array(["a", "b", "c"], dtype=object), np.array([1, 2])]
    new = io._named_grid(io._product_frame(["LAT", "ADM3", "n"], values), {"LAT": "lat"}, "ADM3")
    old = nc_utils._named_grid(nc_utils._product_frame(["LAT", "ADM3", "n"], values), {"LAT": "lat"}, "ADM3")
    pd.testing.assert_frame_equal(new, old)

    days = np.arange(0, 47)
    for options in ({"min_day": 1, "max_day": 45}, {"min_day": 1, "rain_day_max": 46, "rain_horizon_policy": "truncate"}):
        spec = {"options": options}
        for a, b in zip(io._select_forecast_days(days, spec, "f.nc"), nc_utils._select_forecast_days(days, spec, "f.nc")):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="strict rainfall horizon"):
        io._select_forecast_days(days, {"options": {"min_day": 1, "rain_day_max": 40}}, "f.nc")
