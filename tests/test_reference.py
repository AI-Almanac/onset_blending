"""A trained blend's reference data round-trips as arrays, no files."""
import datetime as dt
import io

import numpy as np
import pandas as pd
import pytest

from _legacy import nc_utils as _nc
from onset_blending.core import ids as _ids
from onset_blending.core.reference import Reference


def round_trip(reference: Reference) -> Reference:
    meta, arrays = reference.to_arrays()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return Reference.from_arrays(meta, np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False))


def test_tables_and_scalars_round_trip_exactly():
    reference = Reference(
        thresholds=pd.DataFrame({"id": pd.Series(["Abaala", "Zone 2"], dtype=str), "onset_thresh": [18.5, 20.0]}),
        reference_onset=pd.DataFrame({
            "year": np.array([2020, 2021], dtype=np.int64),
            "ref_onset_date": pd.Series([dt.date(2020, 6, 1), None], dtype=object),
        }),
        cell_transforms={"aifs": pd.DataFrame({
            "source_id": pd.Series(["10.00_70.00", "10.25_70.00"], dtype=str),
            "target_id": pd.Series(["Abaala", "Abaala"], dtype=str), "weight": [0.25, 0.75],
        })},
        regrid_weights={"observations": pd.DataFrame({
            "latitude": [10.0, 10.25], "longitude": [70.0, 70.0],
            "adm3_name": pd.Series(["Abaala", None], dtype=str), "weight": [1.0, 0.5],
        })},
    )
    back = round_trip(reference)
    pd.testing.assert_frame_equal(back.thresholds, reference.thresholds, check_exact=True)
    pd.testing.assert_frame_equal(back.reference_onset, reference.reference_onset, check_exact=True)
    pd.testing.assert_frame_equal(back.cell_transforms["aifs"], reference.cell_transforms["aifs"], check_exact=True)
    pd.testing.assert_frame_equal(
        back.regrid_weights["observations"], reference.regrid_weights["observations"], check_exact=True
    )


def test_a_constant_threshold_and_reference_date_stay_values():
    reference = Reference(thresholds=20.0, reference_onset={"mode": "constant_month_day", "month_day": "06-01"})
    back = round_trip(reference)
    assert back.thresholds == 20.0 and back.reference_onset == reference.reference_onset


# -- lookups: the legacy attach_thresholds_id / attach_ref_onset -------------


_UNITS = np.array(["10.00_70.00", "10.00_70.25", "10.25_70.00", "10.25_70.25"])
_CONVENTION = _ids.GridIdConvention(2, "fixed", "spec")
_SPEC = {"geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}


@pytest.mark.parametrize("thresholds", [
    20.0,
    pd.DataFrame({"lat": ["10.0", "10.0", "10.25", "10.25"], "lon": ["70.0", "70.25", "70.0", "70.25"],
                  "onset_thresh": ["11", "12", "13", "14"]}),
    pd.DataFrame({"id": list(_UNITS[::-1]), "onset_thresh": [1.0, 2.0, 3.0, 4.0]}),
])
def test_unit_thresholds_are_the_legacy_joins(thresholds):
    reference = Reference(thresholds=thresholds, reference_onset=None)

    new = reference.unit_thresholds(_UNITS, _CONVENTION)

    old = _nc.attach_thresholds_id(pd.DataFrame({"id": _UNITS}), thresholds, spec=_SPEC)["onset_thresh"]
    np.testing.assert_array_equal(new, old.to_numpy(dtype=float))


def test_a_threshold_table_must_cover_the_units():
    reference = Reference(thresholds=pd.DataFrame({"id": ["10.00_70.00"], "onset_thresh": [1.0]}), reference_onset=None)
    with pytest.raises(ValueError, match="missing 3 of 4"):
        reference.unit_thresholds(_UNITS, _CONVENTION)


@pytest.mark.parametrize("reference_onset", [
    None,
    {"mode": "constant_month_day", "month_day": "06-01"},
    pd.DataFrame({"year": [2001, 2002], "ref_onset_date": [dt.date(2001, 6, 3), dt.date(2002, 5, 29)]}),
    pd.DataFrame({"id": list(_UNITS), "ref_onset_date": [dt.date(2000, 6, d) for d in (1, 2, 3, 4)]}),
])
def test_reference_dates_are_the_legacy_joins(reference_onset):
    reference = Reference(thresholds=None, reference_onset=reference_onset)
    units, years = np.repeat(_UNITS, 2), np.tile([2001, 2002], len(_UNITS))

    new = reference.reference_dates(units, years)

    old = _nc.attach_ref_onset(pd.DataFrame({"id": units, "year": years}), reference_onset)["ref_onset_date"]
    np.testing.assert_array_equal(new, pd.to_datetime(old).to_numpy().astype("datetime64[ns]"))
