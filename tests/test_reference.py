"""A trained blend's reference data round-trips as arrays, no files."""
import datetime as dt
import io

import numpy as np
import pandas as pd

from onset_blending.reference import Reference


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
