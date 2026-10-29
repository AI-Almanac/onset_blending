import os
import pickle

import numpy as np
import pandas as pd
import pytest

from _legacy.blend_evaluation_utils import (
    ExternalPredictions,
    load_external_predictions,
    parse_external_predictions,
    score_model_predictions,
)
from _legacy.misc import interval_bins

N_BINS = 4
CV_COLS = [f"cv_{b}" for b in interval_bins(N_BINS)]


def make_cv_preds(seed=0, n_cells=6, years=(2001, 2002, 2003), issues_per_year=4):
    rng = np.random.default_rng(seed)
    rows = [
        {"id": f"{10 + c}.0_{70 + c}.0", "year": year, "issue": k}
        for c in range(n_cells)
        for year in years
        for k in range(issues_per_year)
    ]
    frame = pd.DataFrame(rows)
    frame["outcome"] = rng.choice(interval_bins(N_BINS), size=len(frame))
    probs = rng.dirichlet(np.ones(len(CV_COLS)), size=len(frame))
    for j, col in enumerate(CV_COLS):
        frame[col] = probs[:, j]
    frame["p_day_1"] = rng.random(len(frame))
    return frame


def write_pickle(path, obj):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def allowed(frame):
    return pd.DataFrame({"id": sorted(frame["id"].unique())[:-1]})


def test_external_predictions_score_like_an_internal_model(tmp_path):
    cv_preds = make_cv_preds()
    write_pickle(tmp_path / "cv_preds_forest.pkl", cv_preds)
    extras = {"external_predictions": [{"name": "forest", "file": "cv_preds_forest.pkl"}]}

    [external] = parse_external_predictions(extras, str(tmp_path))
    loaded = load_external_predictions(external, N_BINS)
    external_cells, external_yearly = score_model_predictions(
        loaded, external.name, external.method, allowed(cv_preds), N_BINS
    )
    internal_cells, internal_yearly = score_model_predictions(
        cv_preds, "forest", "global", allowed(cv_preds), N_BINS
    )

    pd.testing.assert_frame_equal(external_cells, internal_cells)
    assert external_yearly == internal_yearly
    assert set(external_cells["model"]) == {"forest"}
    assert {row["year"] for row in external_yearly} == {2001, 2002, 2003}


def test_perfect_predictions_score_zero_brier_and_rps():
    cv_preds = make_cv_preds()
    for col in CV_COLS:
        cv_preds[col] = (cv_preds["outcome"] == col[len("cv_"):]).astype(float)

    cells, yearly = score_model_predictions(cv_preds, "oracle", "global", None, N_BINS)

    pooled = cells[cells["id"] == "ALL"].iloc[0]
    assert pooled["brier"] == 0.0 and pooled["rps"] == 0.0
    assert all(row["brier"] == 0.0 and row["rps"] == 0.0 for row in yearly)


def test_cells_outside_the_allowed_set_are_not_scored():
    cv_preds = make_cv_preds()
    allowed_cells = allowed(cv_preds)

    cells, _ = score_model_predictions(cv_preds, "forest", "global", allowed_cells, N_BINS)

    assert set(cells["id"]) - {"ALL"} == set(allowed_cells["id"])


def test_missing_cv_column_is_reported_by_name(tmp_path):
    write_pickle(tmp_path / "preds.pkl", make_cv_preds().drop(columns=["cv_later"]))
    external = ExternalPredictions(name="forest", path=str(tmp_path / "preds.pkl"))

    with pytest.raises(ValueError, match=r"forest.*missing columns \['cv_later'\]"):
        load_external_predictions(external, N_BINS)


def test_week_bins_beyond_the_configured_count_are_rejected(tmp_path):
    write_pickle(tmp_path / "preds.pkl", make_cv_preds().assign(cv_week5=0.0))
    external = ExternalPredictions(name="forest", path=str(tmp_path / "preds.pkl"))

    with pytest.raises(ValueError, match="cv_week5"):
        load_external_predictions(external, N_BINS)


def test_non_dataframe_pickle_is_rejected(tmp_path):
    write_pickle(tmp_path / "preds.pkl", {"cv_week1": [0.1]})
    external = ExternalPredictions(name="forest", path=str(tmp_path / "preds.pkl"))

    with pytest.raises(ValueError, match="must be a pandas DataFrame"):
        load_external_predictions(external, N_BINS)


def test_relative_file_resolves_against_results_dir(tmp_path):
    extras = {"external_predictions": [{"name": "forest", "file": "sub/preds.pkl"}]}

    [external] = parse_external_predictions(extras, str(tmp_path))

    assert external.path == os.path.join(str(tmp_path), "sub", "preds.pkl")
    assert external.method == "global"


def test_absolute_file_is_used_as_is(tmp_path):
    absolute = str(tmp_path / "elsewhere" / "preds.pkl")
    extras = {"external_predictions": [
        {"name": "forest", "file": absolute, "method": "local"},
    ]}

    [external] = parse_external_predictions(extras, "/results")

    assert external == ExternalPredictions(name="forest", path=absolute, method="local")


def test_no_external_predictions_configured():
    assert parse_external_predictions({}, "/results") == []
    assert parse_external_predictions(None, "/results") == []


def test_entry_without_file_is_rejected():
    extras = {"external_predictions": [{"name": "forest"}]}

    with pytest.raises(ValueError, match="missing: file"):
        parse_external_predictions(extras, "/results")
