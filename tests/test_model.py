"""FittedModel fits, converts and predicts exactly as 3_fit_final_model.py and
apply_blend_model.py did."""

import dataclasses
import io
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import patsy
import pytest
from sklearn.preprocessing import StandardScaler

from onset_blending._legacy.blend_evaluation_utils import (
    _make_multinom_clf,
    apply_formula_sample_support,
    build_formulas_from_spec,
    compute_cv_global,
    resolve_climatology_weeks,
    restrict_to_allowed,
)
from onset_blending._legacy.nc_utils import ensure_id_col
from onset_blending._legacy.spatial_id_utils import resolve_grid_id_convention
from onset_blending.evaluation import evaluate
from onset_blending.model import FittedModel, fit_final

BINS = ["week1", "week2", "later"]
FORMULA = "outcome ~ clim_week1 + clim_week2 + x + clim_week1:x"


def _legacy_fit_final_model(formula_str, train_df):
    """3_fit_final_model.py's fit_final_model, as it was (prints removed)."""
    rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    X_df = patsy.dmatrix(
        rhs, train_df, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[])
    )
    X_df = X_df.drop(columns=["Intercept"], errors="ignore")
    feature_cols = list(X_df.columns)
    y = train_df["outcome"].values
    valid = np.array([isinstance(v, str) for v in y])
    X_vals = X_df.values.astype(float)
    mask = valid & np.isfinite(X_vals).all(axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals[mask])
    clf = _make_multinom_clf()
    clf.fit(X_scaled, y[mask])
    rows = []
    for i, cls in enumerate(clf.classes_):
        for j, feat in enumerate(feature_cols):
            rows.append(
                {
                    "class": cls,
                    "feature": feat,
                    "coefficient": float(clf.coef_[i, j]),
                    "intercept": float(clf.intercept_[i]),
                }
            )
    return {
        "coefs": pd.DataFrame(rows),
        "scaler": scaler,
        "features": feature_cols,
        "formula": formula_str,
    }


def _legacy_final_training_rows(wide_df, spec, dissemination_input, model):
    """3_fit_final_model.py's main(): the formula and rows it fitted."""
    training_years = list(map(int, spec["run"]["training_years"]))
    true_holdout_years = set(map(int, spec["run"].get("true_holdout_years") or []))
    fit_years = [year for year in training_years if year not in true_holdout_years]
    wide_df = ensure_id_col(wide_df, spec=spec, context="blend input IDs")
    id_convention = resolve_grid_id_convention(
        spec=spec, authoritative_ids=wide_df["id"], context="blend input/dissemination ID handoff"
    )
    dissemination_cells = ensure_id_col(
        dissemination_input, spec=spec, convention=id_convention, context="blend dissemination IDs"
    ).drop_duplicates("id")
    _, clim_weeks = resolve_climatology_weeks(spec, wide_df)
    formulas = build_formulas_from_spec(
        spec, spec["run"]["cutoff_mode"], data=wide_df, n_bins=max(clim_weeks)
    )
    wide_df, _ = apply_formula_sample_support(wide_df, formulas)
    wide_df_allowed = restrict_to_allowed(wide_df, dissemination_cells)
    return formulas[model], wide_df_allowed[wide_df_allowed["year"].isin(fit_years)].copy()


def _legacy_apply(coef_bundle, test_df):
    """apply_blend_model.py's prediction path (build_application_design_matrix,
    the saved scaler, apply_coefs), as it was."""
    coef_df, scaler, feature_cols = (
        coef_bundle["coefs"],
        coef_bundle["scaler"],
        coef_bundle["features"],
    )
    formula_str = coef_bundle["formula"]
    rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    X_df = (
        patsy.dmatrix(rhs, test_df, return_type="dataframe", NA_action=patsy.NAAction(NA_types=[]))
        .drop(columns=["Intercept"], errors="ignore")
        .loc[:, feature_cols]
    )
    if scaler is not None:
        X_df = pd.DataFrame(
            scaler.transform(X_df.values.astype(float)), columns=feature_cols, index=X_df.index
        )
    classes = sorted(coef_df["class"].unique())
    X = X_df[feature_cols].values.astype(float)
    W = np.zeros((len(classes), len(feature_cols)))
    b = np.zeros(len(classes))
    for i, cls in enumerate(classes):
        cls_rows = coef_df[coef_df["class"] == cls]
        b[i] = cls_rows["intercept"].iloc[0]
        for j, feat in enumerate(feature_cols):
            feat_row = cls_rows[cls_rows["feature"] == feat]
            if not feat_row.empty:
                W[i, j] = float(feat_row["coefficient"].iloc[0])
    scores = X @ W.T + b
    bad = ~np.isfinite(X).all(axis=1)
    probs = np.full((len(X), len(classes)), np.nan)
    if (~bad).any():
        s = scores[~bad]
        s -= s.max(axis=1, keepdims=True)
        exp_s = np.exp(s)
        probs[~bad] = exp_s / exp_s.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, columns=[f"cv_{c}" for c in classes], index=X_df.index)


def weekly_table(seed=0, years=range(2010, 2021), n_cells=4, issues=8):
    rng = np.random.default_rng(seed)
    rows = [
        {"id": f"unit{c}", "year": year, "issue": k}
        for c in range(n_cells)
        for year in years
        for k in range(issues)
    ]
    frame = pd.DataFrame(rows)
    n = len(frame)
    frame["clim_week1"] = rng.normal(-1.0, 0.5, n)
    frame["clim_week2"] = rng.normal(-1.0, 0.5, n)
    frame["x"] = rng.normal(size=n)
    # Outcomes that depend on the features, so the coefficients mean something.
    logits = np.column_stack([frame["clim_week1"] + frame["x"], frame["clim_week2"], np.zeros(n)])
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    frame["outcome"] = [BINS[rng.choice(3, p=row)] for row in p]
    frame.loc[rng.random(n) < 0.05, "outcome"] = np.nan
    frame.loc[5, "x"] = np.nan  # outside the formula sample support
    return frame


SPEC = {
    "run": {
        "cutoff_mode": "fixed_cutoff",
        "training_years": list(range(2010, 2020)),
        "true_holdout_years": [2019],
        "cv_holdout_years": [2018],
    },
    "cell": {},
    "models": {"formulas": {"blend": {"text": "outcome ~ clim_qx + x + clim_week1:x"}}},
    "extras": {"clim_logits": [{"name": "clim_raw", "base_col_prefix": "clim"}]},
}


def dissemination(weekly):
    return pd.DataFrame({"adm3_name": sorted(weekly["id"].unique())[:-1]})


def fitted():
    weekly = weekly_table()
    return weekly, fit_final(weekly, SPEC, dissemination(weekly), model_name="blend")


def test_final_fit_is_the_scripts_bundle():
    weekly, model = fitted()
    formula, train = _legacy_final_training_rows(weekly, SPEC, dissemination(weekly), "blend")
    legacy = _legacy_fit_final_model(formula, train)

    assert model.formula == formula == FORMULA
    assert pickle.dumps(model.to_legacy_bundle()) == pickle.dumps(legacy)
    assert model.classes == ("later", "week1", "week2")
    assert model.n_samples == int(legacy["scaler"].n_samples_seen_)


def test_final_fit_beside_the_cv_folds_is_the_scripts_bundle():
    weekly = weekly_table()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluation = evaluate(weekly, SPEC, dissemination(weekly), workers=2, final_model="blend")
    formula, train = _legacy_final_training_rows(weekly, SPEC, dissemination(weekly), "blend")
    bundle, legacy = evaluation.final_model.to_legacy_bundle(), _legacy_fit_final_model(formula, train)

    # The same content (a worker's strings are its own objects, so the pickle's bytes differ).
    assert sorted(bundle) == sorted(legacy)
    pd.testing.assert_frame_equal(bundle["coefs"], legacy["coefs"], check_exact=True)
    assert (bundle["features"], bundle["formula"]) == (legacy["features"], legacy["formula"])
    assert list(vars(bundle["scaler"])) == list(vars(legacy["scaler"]))
    for name, value in vars(legacy["scaler"]).items():
        np.testing.assert_array_equal(vars(bundle["scaler"])[name], value, strict=True)


def test_legacy_bundle_round_trip():
    _, model = fitted()
    bundle = model.to_legacy_bundle()

    again = FittedModel.from_legacy_bundle(bundle)

    assert pickle.dumps(again.to_legacy_bundle()) == pickle.dumps(bundle)


def test_arrays_round_trip_without_pickles():
    _, model = fitted()
    meta, arrays = model.to_arrays()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    buffer.seek(0)

    with np.load(buffer, allow_pickle=False) as loaded:
        again = FittedModel.from_arrays(json.loads(json.dumps(meta)), dict(loaded))

    assert pickle.dumps(again.to_legacy_bundle()) == pickle.dumps(model.to_legacy_bundle())


def test_predictions_are_apply_blend_models():
    weekly, model = fitted()
    test = weekly_table(seed=1, years=[2021]).set_index(pd.RangeIndex(100, 100 + 4 * 8))
    test.loc[105, "x"] = 0.5
    test.loc[103, "clim_week2"] = np.nan

    predicted = model.predict(test)

    pd.testing.assert_frame_equal(
        predicted, _legacy_apply(model.to_legacy_bundle(), test), check_exact=True
    )
    assert list(predicted.columns) == ["cv_later", "cv_week1", "cv_week2"]
    assert predicted.loc[103].isna().all() and predicted.drop(index=103).notna().all().all()
    np.testing.assert_allclose(predicted.drop(index=103).sum(axis=1), 1.0)


def test_cv_fold_bundles_predict_like_apply_blend_model():
    weekly = weekly_table().dropna(subset=["x"])
    _, coefs = compute_cv_global(
        FORMULA,
        weekly,
        [2018, 2019],
        training_years=range(2010, 2020),
        save_coefs=True,
        required_classes=BINS,
    )
    test = weekly_table(seed=2, years=[2021])

    for bundle in coefs.values():
        pd.testing.assert_frame_equal(
            FittedModel.from_legacy_bundle(bundle).predict(test),
            _legacy_apply(bundle, test),
            check_exact=True,
        )


def test_bundles_without_scaler_or_formula():
    # The bare coefficient tables older runs pickled: unscaled features, no formula.
    _, model = fitted()
    coefs = model.to_legacy_bundle()["coefs"]
    bundle = {"coefs": coefs, "scaler": None, "features": list(coefs["feature"].unique())}
    test = weekly_table(seed=3, years=[2021])

    with pytest.raises(ValueError, match="no saved formula"):
        FittedModel.from_legacy_bundle(bundle)
    unscaled = FittedModel.from_legacy_bundle(bundle, formula=FORMULA)

    assert unscaled.scaler_mean is None and unscaled.to_arrays()[0]["n_samples"] is None
    pd.testing.assert_frame_equal(
        unscaled.predict(test),
        _legacy_apply({**bundle, "formula": FORMULA}, test),
        check_exact=True,
    )


def test_missing_coefficients_count_as_zero_with_a_warning():
    _, model = fitted()
    bundle = model.to_legacy_bundle()
    coefs = bundle["coefs"]
    bundle["coefs"] = coefs[~((coefs["class"] == "week1") & (coefs["feature"] == "x"))]

    with pytest.warns(UserWarning, match="Feature 'x' not found in coefs for class 'week1'"):
        partial = FittedModel.from_legacy_bundle(bundle)

    assert partial.coef[partial.classes.index("week1"), partial.features.index("x")] == 0.0


def test_application_data_must_have_the_saved_features():
    _, model = fitted()
    test = weekly_table(seed=4, years=[2021])

    other_terms = dataclasses.replace(
        model, formula="outcome ~ clim_week1 + clim_week2 + x + clim_week2:x"
    )
    with pytest.raises(ValueError, match="does not match the saved training schema"):
        other_terms.predict(test)
    with pytest.raises(ValueError, match="no finite values"):
        model.predict(test.assign(x=np.nan))


def test_final_fit_requires_every_outcome_class():
    weekly = weekly_table()
    weekly.loc[weekly["outcome"] == "week2", "outcome"] = "later"

    with pytest.raises(ValueError, match="missing outcome classes: week2"):
        fit_final(weekly, SPEC, dissemination(weekly), model_name="blend")
    with pytest.raises(ValueError, match="Model 'other' not found"):
        fit_final(weekly, SPEC, dissemination(weekly), model_name="other")


def test_models_are_immutable():
    _, model = fitted()

    with pytest.raises(ValueError):
        model.coef[0, 0] = 1.0
    with pytest.raises(AttributeError):
        model.formula = "outcome ~ x"
