"""The vectorized paths give exactly what the row-by-row code they replaced gave.

Each test builds the original result the original way (itertools.product,
per-row conversions, row-wise apply, the per-group loop) and requires the
fast path to match it exactly, dtypes included.
"""
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
import pytest

from python.prepare_data import nc_utils
from python.prepare_data.nc_utils import (
    _by_distinct,
    _later_dates,
    _product_frame,
    _row_dates,
    nc_time_to_dates,
    process_ground_truth_rainfall_id,
)
from python.prepare_data.spatial_id_utils import (
    _distinct_stripped_ids,
    normalize_id_series,
    working_copy,
)

ONSET_OPTIONS = {
    "min_day": 1,
    "max_day": 45,
    "window": 3,
    "onset_definition": {
        "wet_day_min_mm": 1.0,
        "follow_days": 21,
        "dry_spell": {"mode": "consecutive_dry", "min_dry_days": 5, "dry_day_min_mm": 1.0},
    },
    "cutoff_month_day": "05-01",
}


def test_product_frame_matches_a_frame_of_product_tuples():
    dims = [
        ("time", np.array([0.0, 6.0, 12.5], dtype=np.float64)),
        ("number", np.arange(3)),
        ("lat", np.array([6.5, 6.75], dtype=np.float32)),
        ("adm3_name", np.array(["Addis", " Bahir ", "Gondar"], dtype=object)),
        ("flag", np.array([True, False])),
    ]
    names = [name for name, _ in dims]
    values = [vals for _, vals in dims]

    expected = pd.DataFrame(list(product(*values)), columns=names)

    pd.testing.assert_frame_equal(_product_frame(names, values), expected, check_exact=True)


def test_product_frame_of_an_empty_dimension_matches_too():
    values = [np.array([1.0, 2.0]), np.array([], dtype=float)]
    expected = pd.DataFrame(list(product(*values)), columns=["a", "b"])
    pd.testing.assert_frame_equal(_product_frame(["a", "b"], values), expected)


def test_row_dates_match_converting_every_row():
    units = "hours since 1900-01-01 00:00:00"
    rows = np.repeat(np.array([1_000_000.0, 1_000_024.0, 1_000_048.0]), 4)
    dates = nc_time_to_dates(rows, units)

    got_dates, got_years = _row_dates(rows, units)

    assert list(got_dates) == list(dates.date)
    assert got_dates.dtype == dates.date.dtype
    np.testing.assert_array_equal(got_years, pd.DatetimeIndex(dates).year.to_numpy())


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(np.array([" 7.25_38.5", "7.25_38.5", "9_40 "] * 50, dtype=object), name="id"),
        pd.Series(["a", "b", "a"]).astype(str),
        pd.Series(["a", " b"], dtype="string"),
        pd.Series(np.arange(6) % 4),
        pd.Series([1, 1.0, "1", True], dtype=object),
        pd.Series(pd.Categorical(["x", "y", "x"])),
    ],
)
def test_normalize_id_series_matches_normalizing_every_row(values):
    expected = values.astype("string").str.strip().astype(str)
    pd.testing.assert_series_equal(normalize_id_series(values), expected, check_exact=True)
    assert _distinct_stripped_ids(values) == (
        pd.Series(values, dtype="string").dropna().str.strip().unique().tolist()
    )


@pytest.mark.parametrize("values", [pd.Series(["a", None]), pd.Series(["a", "  "])])
def test_normalize_id_series_still_rejects_missing_or_empty_ids(values):
    with pytest.raises(ValueError):
        normalize_id_series(values)


def test_by_distinct_and_later_dates_match_row_wise_apply():
    years = pd.Series([2001, 2002, 2001, 2003] * 10, index=np.arange(40) * 2)
    expected = years.apply(lambda y: pd.Timestamp(f"{y}-05-01").date())
    got = _by_distinct(years, lambda y: y.apply(lambda y: pd.Timestamp(f"{y}-05-01").date()))
    pd.testing.assert_series_equal(got, expected)

    frame = pd.DataFrame({"cutoff_date": expected})
    frame["ref_onset_date"] = [date(2001, 6, 1), date(2002, 4, 1)] * 20
    by_row = frame.apply(lambda r: max(r["cutoff_date"], r["ref_onset_date"]), axis=1)
    assert list(_later_dates(frame["cutoff_date"], frame["ref_onset_date"])) == list(by_row)


def test_working_copy_leaves_the_original_untouched():
    original = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    snapshot = original.copy()
    copy = working_copy(original)
    copy["a"] = copy["a"] * 10
    copy.loc[0, "b"] = "changed"
    copy["c"] = 1
    pd.testing.assert_frame_equal(original, snapshot)


def _ground_truth(seed=0, n_cells=5, years=(2001, 2002, 2003)):
    rng = np.random.default_rng(seed)
    rows = []
    for cell, year in product(range(n_cells), years):
        days = pd.date_range(f"{year}-04-01", f"{year}-09-30", freq="D")
        rain = rng.gamma(0.6, 6.0, len(days))
        rain[rng.random(len(days)) < 0.05] = np.nan
        if cell == 0 and year == years[0]:
            rain[:] = np.nan  # a series with no data
        for day, value in zip(days, rain):
            rows.append({"id": f" cell{cell}", "time": day.date(), "year": year, "rainfall": value})
    frame = pd.DataFrame(rows)
    # The readers hand the rows over unsorted by cell.
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize(
    "ref_onset",
    [None, {"mode": "constant_month_day", "month_day": "06-01"}],
)
def test_ground_truth_onsets_match_the_per_group_loop(ref_onset):
    frame = _ground_truth()
    spec = {"options": ONSET_OPTIONS, "input": {"value_col": "RAINFALL"}}

    fast = process_ground_truth_rainfall_id(
        frame, {**spec, "output": {"write_long": False}},
        ref_onset_dt=ref_onset, thr_dt=20.0, value_col="rainfall",
    )
    # write_long keeps the original groupby loop, which builds the same wide table.
    loop = process_ground_truth_rainfall_id(
        frame, {**spec, "output": {"write_long": True}},
        ref_onset_dt=ref_onset, thr_dt=20.0, value_col="rainfall",
    )

    assert fast["long"] is None
    pd.testing.assert_frame_equal(fast["wide"], loop["wide"], check_exact=True)
    assert fast["wide"]["onset_day"].notna().any()


def test_forecast_start_days_match_one_cutoff_per_row():
    issue = pd.to_datetime(["2001-04-20", "2002-05-03", "2001-05-10"]).values
    years = np.array([2001, 2002, 2001])
    day_ints = np.arange(1, 46)
    ref = np.array(["NaT", "2002-06-01", "2001-06-01"], dtype="datetime64[ns]")

    got = nc_utils._forecast_start_days(issue, years, ref, day_ints, "05-01")

    cutoff = np.array([np.datetime64(f"{y}-05-01") for y in years])
    offsets = (cutoff - issue).astype("timedelta64[D]").astype(int)
    fixed = np.searchsorted(day_ints, offsets - 1, side="right") + 1
    np.testing.assert_array_equal(got[0], fixed)
