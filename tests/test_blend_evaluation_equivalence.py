"""The faster blend-evaluation paths give exactly what the code they replaced gave."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from python.blending_process.blend_evaluation_utils import (
    _fit_columns,
    compute_cv_global,
    platt_cv_multibin,
)

BINS = ["week1", "week2", "later"]


def _original_platt_cv_multibin(df, prob_cols, holdout_years, true_holdout_years=(),
                                outcome_col="outcome", year_col="year",
                                cv_prefix="cv_", allowed_cells=None):
    """platt_cv_multibin as it was, filtering and copying the table per fold."""
    bins = prob_cols
    out_list = []
    for test_year in holdout_years:
        train = df[(df[year_col] != test_year) & (~df[year_col].isin(true_holdout_years))].copy()
        test = df[df[year_col] == test_year].copy()
        if test.empty:
            continue
        if allowed_cells is not None:
            allowed_ids = set(allowed_cells["id"])
            train = train[train["id"].isin(allowed_ids)]
        cal_mat = np.full((len(test), len(bins)), np.nan)
        for j, b in enumerate(bins):
            p_tr = np.clip(train[b].values.astype(float), 1e-6, 1 - 1e-6)
            outcome = train[outcome_col]
            y_tr = ((outcome == b).astype(float)
                    .where(outcome.notna(), np.nan).to_numpy())
            ok = np.isfinite(p_tr) & ~np.isnan(y_tr)
            if ok.sum() < 10 or len(np.unique(y_tr[ok])) < 2:
                cal_mat[:, j] = test[b].values.astype(float)
                continue
            X = p_tr[ok].reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
            clf.fit(X, y_tr[ok])
            test_p = np.clip(test[b].values.astype(float), 1e-6, 1 - 1e-6).reshape(-1, 1)
            cal_mat[:, j] = expit(clf.intercept_[0] + clf.coef_[0][0] * test_p.flatten())
        rs = cal_mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        cal_mat /= rs
        test_out = test.copy()
        for j, b in enumerate(bins):
            test_out[f"{cv_prefix}{b}"] = cal_mat[:, j]
        out_list.append(test_out)
    if out_list:
        return pd.concat(out_list, ignore_index=True)
    return pd.DataFrame()


def _table(seed=0, n_cells=8, years=range(2001, 2009), rows_per_cell_year=6):
    rng = np.random.default_rng(seed)
    rows = [
        {"id": f"c{cell}", "year": year, "issue": k}
        for cell in range(n_cells) for year in years for k in range(rows_per_cell_year)
    ]
    frame = pd.DataFrame(rows)
    probs = rng.dirichlet(np.ones(len(BINS)), size=len(frame))
    for j, b in enumerate(BINS):
        frame[b] = probs[:, j]
    frame["x1"] = rng.normal(size=len(frame))
    frame["x2"] = rng.normal(size=len(frame))
    frame["outcome"] = rng.choice(BINS, size=len(frame))
    frame.loc[rng.random(len(frame)) < 0.05, "outcome"] = np.nan
    frame["unused"] = "text"
    return frame


@pytest.mark.parametrize("allowed", [None, pd.DataFrame({"id": ["c0", "c2", "c3", "c5"]})])
def test_platt_cv_matches_filtering_the_table_per_fold(allowed):
    frame = _table()
    args = dict(prob_cols=BINS, holdout_years=[2003, 2004, 2007],
                true_holdout_years=(2008,), allowed_cells=allowed)

    pd.testing.assert_frame_equal(
        platt_cv_multibin(frame, **args), _original_platt_cv_multibin(frame, **args),
        check_exact=True,
    )


def test_global_cv_in_worker_processes_matches_the_serial_folds():
    frame = _table(seed=1)
    formula = "outcome ~ x1 + x2 + x1:x2"
    args = dict(holdout_years=[2002, 2005, 2006], training_years=list(range(2001, 2008)),
                true_holdout_years=(2008,), save_coefs=True, required_classes=BINS)

    serial, serial_coefs = compute_cv_global(formula, frame, data_pred=frame, n_jobs=1, **args)
    pooled, pooled_coefs = compute_cv_global(formula, frame, data_pred=frame, n_jobs=3, **args)

    pd.testing.assert_frame_equal(pooled, serial, check_exact=True)
    # The fold rows carry every column of the prediction table, not just the fit's.
    assert {"unused", "issue", *BINS} <= set(pooled.columns)
    for year in args["holdout_years"]:
        pd.testing.assert_frame_equal(pooled_coefs[year]["coefs"], serial_coefs[year]["coefs"])


def test_fit_columns_fall_back_to_the_whole_frame_for_non_column_terms():
    frame = _table()
    assert _fit_columns(frame, "outcome ~ x1 + x1:x2", ["year", "outcome"]) == [
        "x1", "x2", "year", "outcome",
    ]
    assert _fit_columns(frame, "outcome ~ np.log(x1) + x2", ["year"]) is None
