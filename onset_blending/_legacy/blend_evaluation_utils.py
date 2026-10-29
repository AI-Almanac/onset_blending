# ==============================================================================
# File: blend_evaluation_utils.py
# ==============================================================================
# Purpose
#   Utility functions for the 2025 weekly-bin blending evaluation pipeline.
#   Provides spec-driven formula building, raw/calibrated prediction helpers,
#   Platt scaling, blend weight optimization, metric computation, and mapping
#   utilities.
# ==============================================================================

import re
import os
from dataclasses import dataclass
import patsy
import pickle
import tempfile
import warnings
import numpy as np
import pandas as pd
from multiprocessing import get_context, shared_memory
from scipy.special import logit, expit
from scipy.stats import rankdata
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .misc import coalesce, week_labels, interval_bins


DEFAULT_PIPELINE_INPUT_ROOT = os.path.join(
    "Monsoon_Data", "Processed_Data", "pipeline_input"
)


def _present_weeks(df, base):
    """Sorted week numbers present as `<base>_week<n>` columns in df."""
    pat = re.compile(rf"^{re.escape(base)}_week(\d+)$")
    return sorted(int(m.group(1)) for c in df.columns for m in [pat.match(c)] if m)


def resolve_climatology_weeks(spec, df):
    """Return the configured conditional-climatology prefix and its weeks."""
    clim_tasks = ((spec.get("extras") or {}).get("clim_logits") or [])
    clim_task = next(
        (task for task in clim_tasks if task.get("name") == "clim_raw"),
        {},
    )
    base = coalesce(clim_task.get("base_col_prefix"), "prob_clim")
    weeks = _present_weeks(df, base)
    if not weeks:
        raise ValueError(
            f"No climatology columns found for configured prefix '{base}' "
            f"(expected {base}_week<n>)."
        )
    return base, weeks


def _cv_weeks(df):
    """Sorted week numbers present as `cv_week<n>` columns in df."""
    return sorted(int(m.group(1)) for c in df.columns
                  for m in [re.match(r"^cv_week(\d+)$", c)] if m)


def _spec_n_bins(spec):
    """Number of forecast bins from the spec (default 4)."""
    return int(coalesce((spec or {}).get("n_bins"), 4))

# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def get_forecast_variant_suffix(spec, variant):
    """Map variant name to column suffix string."""
    m = (spec.get("extras") or {}).get("forecast_variants") or {
        "base": "",
        "ref_onset": "_ref",
        "fixed_cutoff": "_fixed_cutoff",
    }
    if variant not in m:
        raise ValueError(f"Unknown forecast variant: {variant}")
    return m[variant]


def forecast_prob_cols(forecast_name, variant_suffix, n_bins=4):
    """Build dict of week1..weekN probability column names."""
    base = f"{forecast_name}_p_onset{variant_suffix}"
    return {f"week{w}": f"{base}_week{w}" for w in range(1, int(n_bins) + 1)}


def forecast_label(forecast_name, variant):
    return forecast_name if variant == "base" else f"{forecast_name}_{variant}"


def make_year_tag(years):
    years = sorted(set(years))
    if not years:
        return ""
    if len(years) == 1:
        return f"_{years[0]}"
    return f"_{min(years)}_{max(years)}"


def configured_holdout_years(spec):
    """Return true and CV holdout years under the shared artifact contract."""
    run = (spec or {}).get("run") or {}
    true_holdout = list(map(int, coalesce(run.get("true_holdout_years"), [])))
    cv_holdout = list(map(int, coalesce(run.get("cv_holdout_years"), [])))
    return true_holdout + cv_holdout


def make_cutoff_tag(cutoff_mode):
    if cutoff_mode == "fixed_cutoff":
        return "_fixed_cutoff"
    if cutoff_mode == "no_ref_filter":
        return "_no_ref_filter"
    return ""


def input_rds_from_cutoff(cutoff_mode, resolution=""):
    # Derive the CV-input filename from the single-source-of-truth cutoff tag
    # (make_cutoff_tag) instead of hardcoding the variant token. The variant
    # segment is generic (e.g. "fixed_cutoff"), so this tracks any rename.
    prefix = f"{resolution}_" if resolution else ""
    variant = make_cutoff_tag(cutoff_mode).lstrip("_")
    variant_part = f"{variant}_" if variant else ""
    return f"cv_data_{prefix}{variant_part}new_pipeline.pkl"


def resolve_blend_input_path(
    cutoff_mode,
    run_cfg=None,
    *,
    input_path=None,
    work_dir=None,
    pipeline_input_root=DEFAULT_PIPELINE_INPUT_ROOT,
):
    """Resolve a connector artifact for evaluation or saved-model use."""
    if input_path:
        return input_path

    run_cfg = run_cfg or {}
    if "input_rds_override" in run_cfg:
        raise ValueError(
            "run.input_rds_override has been removed; use --input_path "
            "or --blend_input."
        )

    input_file = input_rds_from_cutoff(cutoff_mode)
    if work_dir:
        return os.path.join(work_dir, input_file)

    return os.path.join(
        pipeline_input_root,
        run_cfg.get("pipeline_input_dir", ""),
        input_file,
    )


# ---------------------------------------------------------------------------
# Formula expansion
# ---------------------------------------------------------------------------

#def expand_formula_str(formula_str):
#    """
#    Expand formula shortcuts in a formula string.
#    Terms containing '_qx' expand to '_week1'..'_week4'.
#    Returns the expanded formula string.
#    """
#    if "~" in formula_str:
#        lhs, rhs = formula_str.split("~", 1)
#        lhs, rhs = lhs.strip(), rhs.strip()
#    else:
#        lhs, rhs = "", formula_str.strip()
#
#    terms = [t.strip() for t in rhs.split("+")]
#    expanded = []
#    for term in terms:
#        if "_qx" in term:
#            for i in range(1, 5):
#                expanded.append(term.replace("_qx", f"_week{i}"))
#        else:
#            expanded.append(term)
#
#    new_rhs = " + ".join(expanded)
#    return f"{lhs} ~ {new_rhs}" if lhs else new_rhs


# AFTER
def _expanded_formula_terms(formula_str, n_bins=4):
    """Return ``(lhs, [(term, qx_generated_columns), ...])``."""
    import itertools

    if "~" in formula_str:
        lhs, rhs = formula_str.split("~", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
    else:
        lhs, rhs = "", formula_str.strip()

    qx_expanded = []
    for raw_term in (t.strip() for t in rhs.split("+")):
        qx_columns = [
            column for column in _parse_formula_cols(raw_term)
            if "_qx" in column
        ]
        if qx_columns:
            for i in range(1, int(n_bins) + 1):
                qx_expanded.append((
                    raw_term.replace("_qx", f"_week{i}"),
                    {
                        column.replace("_qx", f"_week{i}")
                        for column in qx_columns
                    },
                ))
        else:
            qx_expanded.append((raw_term, set()))

    final_terms = []
    for term, qx_generated_columns in qx_expanded:
        if "*" in term:
            variables = [value.strip() for value in term.split("*")]
            for size in range(1, len(variables) + 1):
                for combo in itertools.combinations(variables, size):
                    final_terms.append((
                        ":".join(combo),
                        qx_generated_columns.intersection(combo),
                    ))
        else:
            final_terms.append((term, qx_generated_columns))
    return lhs, final_terms


def _join_formula_terms(lhs, terms):
    rhs = " + ".join(terms)
    return f"{lhs} ~ {rhs}" if lhs else rhs


def expand_formula_str(formula_str, n_bins=4):
    """
    Expand formula shortcuts in a formula string.
    1. Terms containing '_qx' expand to '_week1'..'_weekN' (N = n_bins).
    2. '*' terms are expanded R/Wilkinson-style: a*b*c -> a + b + c + a:b + a:c + b:c + a:b:c
    Returns the expanded formula string.
    """
    lhs, terms = _expanded_formula_terms(formula_str, n_bins=n_bins)
    return _join_formula_terms(lhs, [term for term, _ in terms])


def resolve_formula_str(formula_str, data, n_bins=4):
    """Expand a formula and omit only unavailable terms generated by ``_qx``.

    Structural unavailability comes exclusively from connector metadata. An
    explicitly named ``_weekN`` term is never omitted and remains subject to
    normal feature validation.
    """
    unavailable = set(data.attrs.get("unavailable_rain_predictors", {}))
    lhs, expanded = _expanded_formula_terms(formula_str, n_bins=n_bins)
    kept = []
    dropped = []
    for term, qx_generated_columns in expanded:
        if qx_generated_columns.intersection(unavailable):
            dropped.append(term)
        else:
            kept.append(term)

    if not any(_parse_formula_cols(term) for term in kept):
        raise ValueError(
            "Formula has no constructible predictor terms after resolving "
            "connector rainfall horizons."
        )
    return _join_formula_terms(lhs, kept), dropped


def make_window_suffix(start_year, end_year):
    return f"_{start_year}_{end_year}"


def build_formulas_from_spec(spec, cutoff_mode, data=None,
                             return_resolution=False, n_bins=None):
    """
    Build dict of formula strings from spec["models"]["formulas"],
    with optional windowed variants.
    """
    formula_cfg = (spec.get("models") or {}).get("formulas") or {}
    if not formula_cfg:
        raise ValueError("Spec must define models.formulas with named entries containing 'text'.")

    n_bins = _spec_n_bins(spec) if n_bins is None else int(n_bins)
    base_texts = {}
    resolution = {}
    for nm, v in formula_cfg.items():
        if v.get("enabled", True):
            txt = v.get("text")
            if not txt:
                raise ValueError(f"Formula '{nm}' is enabled but has no non-empty 'text' in spec.")
            expanded = expand_formula_str(txt, n_bins=n_bins)
            if data is None:
                resolved, dropped = expanded, []
            else:
                resolved, dropped = resolve_formula_str(
                    txt, data, n_bins=n_bins
                )
            base_texts[nm] = resolved
            resolution[nm] = {
                "original_formula": txt,
                "expanded_formula": expanded,
                "resolved_formula": resolved,
                "dropped_terms": dropped,
            }

    formulas = dict(base_texts)

    wcfg_all = (spec.get("models") or {}).get("window_variants")
    if wcfg_all:
        wcfg_list = wcfg_all if isinstance(wcfg_all, list) else [wcfg_all]
        for wcfg in wcfg_list:
            if not wcfg.get("enabled", False):
                continue
            only_if = wcfg.get("only_if_cutoff_mode")
            if only_if and cutoff_mode != only_if:
                continue
            base_name = wcfg.get("base_name")
            if not base_name or base_name not in base_texts:
                raise ValueError("window_variants.base_name must match a name in models.formulas.")
            start_years = [int(y) for y in (wcfg.get("start_years") or [])]
            end_year = int(wcfg.get("end_year", 0))
            if not start_years or not end_year:
                raise ValueError("window_variants must define start_years and end_year.")
            from_pat = wcfg.get("replace", {}).get("from")
            to_pat = wcfg.get("replace", {}).get("to")
            if not from_pat or not to_pat:
                raise ValueError("window_variants.replace must define 'from' and 'to'.")

            base_txt = base_texts[base_name]
            for sy in start_years:
                if sy == 1900:
                    continue
                window = make_window_suffix(sy, end_year)
                to_replaced = to_pat.replace("{window}", window)
                windowed = base_txt.replace(from_pat, to_replaced)
                model_name = f"{base_name}{window}"
                formulas[model_name] = expand_formula_str(
                    windowed, n_bins=n_bins
                )
                resolution[model_name] = {
                    "original_formula": windowed,
                    "expanded_formula": windowed,
                    "resolved_formula": formulas[model_name],
                    "dropped_terms": resolution[base_name]["dropped_terms"],
                }

    if return_resolution:
        return formulas, resolution
    return formulas


def print_formula_summary(spec, cutoff_mode, formulas=None):
    """
    Print a human-readable summary of all blending models: the yml shorthand,
    the fully expanded formula, and the exact list of predictor columns used.

    Call this at the start of 1_blend_evaluation.py to make the models
    transparent before any fitting happens.
    """
    if formulas is None:
        formulas = build_formulas_from_spec(spec, cutoff_mode)
    formula_cfg = (spec.get("models") or {}).get("formulas") or {}

    width = 70
    print()
    print("=" * width)
    print("  BLENDING MODELS — formula resolution summary")
    print("=" * width)

    for model_name, expanded in formulas.items():
        yml_text = None
        if model_name in formula_cfg:
            yml_text = formula_cfg[model_name].get("text")

        feature_cols = _parse_formula_cols(expanded)

        print()
        print(f"  Model: {model_name}")
        print(f"  {'-' * (width - 2)}")

        if yml_text:
            print(f"  yml shorthand:")
            print(f"    {yml_text}")
            print()

        print(f"  Effective formula:")
        lhs, rhs = expanded.split("~", 1)
        terms = [t.strip() for t in rhs.split("+")]
        lines = []
        line = ""
        for i, t in enumerate(terms):
            sep = " + " if i < len(terms) - 1 else ""
            if len(line) + len(t) + len(sep) > 60 and line:
                lines.append(line)
                line = t + sep
            else:
                line += t + sep
        if line:
            lines.append(line)
        indent = "    " + " " * (len(lhs.strip()) + 3)
        print(f"    {lhs.strip()} ~  {lines[0]}")
        for l in lines[1:]:
            print(f"    {indent}{l}")

        print()

# BEFORE
        #print(f"  Predictor columns used ({len(feature_cols)}):")
        #groups = {}
        #for col in feature_cols:
        #    root = re.sub(r"_week\d$", "", col)
        #    groups.setdefault(root, []).append(col)
        #for root, cols in groups.items():
        #    weeks = ", ".join(c.split("_week")[-1] for c in cols)
        #    scale = "(logit)" if "clim" in root else "(mm)"
        #    print(f"    {root}_week[{weeks}]  {scale}")

# AFTER
        lhs_exp, rhs_exp = expanded.split("~", 1)
        all_terms = [t.strip() for t in rhs_exp.split("+")]
        print(f"  Predictor terms used ({len(all_terms)}):")
        for term in all_terms:
            scale = "(logit)" if "clim" in term else ("(interaction)" if ":" in term else "(mm)")
            print(f"    {term}  {scale}")

    print()
    print("=" * width)
    print(f"  Total models: {len(formulas)}")
    print("=" * width)
    print()

# ---------------------------------------------------------------------------
# Raw and calibrated predictions
# ---------------------------------------------------------------------------

def make_raw_preds_from_wide(wide_df, forecast_name, variant, holdout_years, spec):
    """Extract raw model predictions for holdout years (any number of week bins)."""
    suf = get_forecast_variant_suffix(spec, variant)
    base = f"{forecast_name}_p_onset{suf}"
    weeks = _present_weeks(wide_df, base)
    if not weeks:
        warnings.warn(f"Skipping raw for {forecast_name} ({variant}); no {base}_week* columns found")
        return None

    sub = wide_df[wide_df["year"].isin(holdout_years)].copy()
    cv_cols = []
    for w in weeks:
        sub[f"cv_week{w}"] = sub[f"{base}_week{w}"]
        cv_cols.append(f"cv_week{w}")
    sub["cv_later"] = np.clip(1.0 - sub[cv_cols].sum(axis=1), 0.0, 1.0)
    return sub[["outcome", "time", "id", "year"] + cv_cols + ["cv_later"]]


def make_raw_preds_from_wide_logit_window(wide_df, base_col_prefix, holdout_years,
                                           start_year=None, end_year=None):
    """Extract raw clim-logit predictions (with optional window suffix)."""
    if start_year is not None and end_year is not None and start_year != 1900:
        window = f"_{start_year}_{end_year}"
    else:
        window = ""
    prefix = f"{base_col_prefix}{window}"
    weeks = _present_weeks(wide_df, prefix)
    if not weeks:
        warnings.warn(f"Skipping clim logit raw for prefix={prefix}; no _week* columns found")
        return None

    sub = wide_df[wide_df["year"].isin(holdout_years)].copy()
    cv_cols = []
    for w in weeks:
        sub[f"cv_week{w}"] = expit(sub[f"{prefix}_week{w}"])
        cv_cols.append(f"cv_week{w}")
    sub["cv_later"] = np.clip(1.0 - sub[cv_cols].sum(axis=1), 0.0, 1.0)
    return sub[["outcome", "time", "id", "year"] + cv_cols + ["cv_later"]]


def make_raw_preds_from_wide_logit(wide_df, base_col_prefix, holdout_years,
                                    earlier_col=None, earlier_is_logit=True,
                                    add_cv_earlier=True, renormalize_6=True):
    """Extract raw clim-logit predictions with optional 'earlier' bin (any n bins)."""
    weeks = _present_weeks(wide_df, base_col_prefix)
    if not weeks:
        warnings.warn(f"Skipping raw-logit for {base_col_prefix}; no _week* columns found")
        return None
    if add_cv_earlier and (not earlier_col or earlier_col not in wide_df.columns):
        raise ValueError(f"unc_clim_raw requires earlier_col and it must exist in wide_df.")

    sub = wide_df[wide_df["year"].isin(holdout_years)].copy()
    week_cols = []
    for w in weeks:
        sub[f"cv_week{w}"] = expit(sub[f"{base_col_prefix}_week{w}"])
        week_cols.append(f"cv_week{w}")

    if add_cv_earlier:
        pE = expit(sub[earlier_col]) if earlier_is_logit else sub[earlier_col].astype(float)
        sub["cv_earlier"] = np.clip(pE, 0.0, 1.0)
    else:
        sub["cv_earlier"] = 0.0

    sub["cv_later"] = np.clip(1.0 - (sub["cv_earlier"] + sub[week_cols].sum(axis=1)), 0.0, 1.0)

    if renormalize_6 and add_cv_earlier:
        allc = ["cv_earlier"] + week_cols + ["cv_later"]
        rs = sub[allc].sum(axis=1)
        good = rs.notna() & (rs > 0)
        for c in allc:
            sub.loc[good, c] = sub.loc[good, c] / rs[good]

    cols = ["outcome", "time", "id", "year"] + week_cols + ["cv_later"]
    if add_cv_earlier:
        cols.append("cv_earlier")
    return sub[cols]


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def fit_platt_weights_export(df, prob_cols, training_years,
                              outcome_col="outcome", year_col="year"):
    """Fit one-vs-rest Platt calibration weights on training years.

    Exported weights are consumed as expit(intercept + slope * logit(p)) by
    apply_platt_5, matching R's glm(y ~ logit_p, ...) export/application pair.
    """
    df_train = df[df[year_col].isin(training_years)].copy()
    weights_rows = []

    for bin_name in prob_cols:
        p_raw = np.clip(df_train[bin_name].values.astype(float), 1e-6, 1 - 1e-6)
        outcome = df_train[outcome_col]
        y = ((outcome == bin_name).astype(float)
             .where(outcome.notna(), np.nan).to_numpy())

        ok = np.isfinite(p_raw) & ~np.isnan(y)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            weights_rows.append({"bin": bin_name, "intercept": 0.0, "slope": 1.0})
            continue

        X = logit(p_raw[ok]).reshape(-1, 1)
        clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
        clf.fit(X, y[ok])
        weights_rows.append({
            "bin": bin_name,
            "intercept": float(clf.intercept_[0]),
            "slope": float(clf.coef_[0][0]),
        })

    weights_df = pd.DataFrame(weights_rows)
    return {"weights_df": weights_df}


def platt_cv_multibin(df, prob_cols, holdout_years, true_holdout_years=(),
                       outcome_col="outcome", year_col="year",
                       cv_prefix="cv_", allowed_cells=None, executor=None):
    """Cross-validated per-bin Platt calibration.

    IMPORTANT: matches R glm(y ~ p, ...) which regresses on raw probability p,
    NOT logit(p). This produces different calibrated values than the logit version.

    Each fit runs with one BLAS thread, so the results do not depend on the
    machine's core count; with an executor (spawned, process-based) the
    holdout years are fitted in parallel, with the same results.
    """
    bins = prob_cols
    out_list = []

    # Per-row inputs every fold reads, computed once (each fold used to filter
    # and copy the whole table and compare every outcome to every bin): a
    # fold's training rows are then positions into these arrays.
    years = df[year_col].to_numpy()
    outside_true_holdout = ~df[year_col].isin(true_holdout_years).to_numpy()
    if allowed_cells is not None:
        allowed = df["id"].isin(set(allowed_cells["id"])).to_numpy()
    else:
        allowed = np.ones(len(df), dtype=bool)
    outcome = df[outcome_col]
    outcome_known = outcome.notna()
    p_all = {b: np.clip(df[b].values.astype(float), 1e-6, 1 - 1e-6) for b in bins}
    y_all = {
        b: (outcome == b).astype(float).where(outcome_known, np.nan).to_numpy()
        for b in bins
    }

    arrays = {
        "p": np.stack([p_all[b] for b in bins]),
        "y": np.stack([y_all[b] for b in bins]),
        "years": years,
        "trainable": outside_true_holdout & allowed,
    }
    if executor is None:
        folds = [_platt_fold(arrays, test_year) for test_year in holdout_years]
    else:
        from .._parallel import SharedArrays

        with SharedArrays(**arrays) as shared:
            futures = [executor.submit(_platt_fold, shared.specs, test_year) for test_year in holdout_years]
            folds = [future.result() for future in futures]

    for test_year, fold in zip(holdout_years, folds):
        test_rows = years == test_year
        test = df[test_rows].copy()
        if test.empty:
            continue
        cal_mat, fallback = fold
        for j in fallback:
            cal_mat[:, j] = test[bins[j]].values.astype(float)

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


def _platt_fold(arrays, test_year):
    """One holdout year's per-bin Platt fits: (calibrated test probabilities,
    bins left uncalibrated). arrays holds p, y (bin, row), years and
    trainable, or their shared-memory specs."""
    from threadpoolctl import threadpool_limits

    if not isinstance(arrays["p"], np.ndarray):
        from .._parallel import attach

        arrays = attach(arrays)
    p_all, y_all, years = arrays["p"], arrays["y"], arrays["years"]
    test_rows = years == test_year
    train_positions = np.flatnonzero((years != test_year) & arrays["trainable"])
    cal_mat = np.full((int(test_rows.sum()), len(p_all)), np.nan)
    fallback = []
    with threadpool_limits(limits=1):
        for j in range(len(p_all)):
            p_tr = p_all[j][train_positions]
            y_tr = y_all[j][train_positions]
            ok = np.isfinite(p_tr) & ~np.isnan(y_tr)
            if ok.sum() < 10 or len(np.unique(y_tr[ok])) < 2:
                fallback.append(j)
                continue

            # Match R: use raw p as the predictor (not logit)
            X = p_tr[ok].reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
            clf.fit(X, y_tr[ok])

            # Predict on test using raw probability
            test_p = p_all[j][test_rows].reshape(-1, 1)
            cal_mat[:, j] = expit(clf.intercept_[0] + clf.coef_[0][0] * test_p.flatten())
    return cal_mat, fallback


def make_calibrated_preds_from_wide(wide_df, forecast_name, variant,
                                    training_years, holdout_years, true_holdout_years,
                                    allowed_cells, spec, executor=None):
    """Platt-calibrated predictions via platt_cv_multibin (any number of week bins)."""
    suf = get_forecast_variant_suffix(spec, variant)
    base = f"{forecast_name}_p_onset{suf}"
    weeks = _present_weeks(wide_df, base)
    if not weeks:
        warnings.warn(f"Skipping calibrated for {forecast_name} ({variant}); no {base}_week* columns found")
        return None

    df = wide_df[wide_df["year"].isin(list(training_years) + list(holdout_years))].copy()
    wk_labels = [f"week{w}" for w in weeks]
    for w in weeks:
        df[f"week{w}"] = df[f"{base}_week{w}"]
    df["later"] = np.maximum(0.0, 1.0 - df[wk_labels].sum(axis=1))

    return platt_cv_multibin(
        df,
        prob_cols=wk_labels + ["later"],
        holdout_years=holdout_years,
        true_holdout_years=true_holdout_years,
        outcome_col="outcome",
        year_col="year",
        cv_prefix="cv_",
        allowed_cells=allowed_cells,
        executor=executor,
    )


# ---------------------------------------------------------------------------
# Multinomial CV
# ---------------------------------------------------------------------------
import itertools

def expand_wilkinson(formula_str, df):
    """
    Expand R-style Wilkinson formula: a*b*c -> a + b + c + a:b + a:c + b:c + a:b:c
    Returns (feature_cols, X) where X is the design matrix as a DataFrame.
    """
    if "~" in formula_str:
        rhs = formula_str.split("~", 1)[1].strip()
    else:
        rhs = formula_str.strip()

    # Split top-level terms on '+'
    top_terms = [t.strip() for t in rhs.split("+")]

    all_cols = {}
    for term in top_terms:
        if "*" in term:
            # Get the base variable names
            vars_ = [v.strip() for v in term.split("*")]
            # Generate all subsets of size 1..n
            for r in range(1, len(vars_) + 1):
                for combo in itertools.combinations(vars_, r):
                    col_name = ":".join(combo)
                    if len(combo) == 1:
                        all_cols[col_name] = df[combo[0]]
                    else:
                        # Product of all variables in combo
                        product = df[combo[0]].copy()
                        for v in combo[1:]:
                            product = product * df[v]
                        all_cols[col_name] = product
        else:
            # Plain additive term
            if term in df.columns:
                all_cols[term] = df[term]

    X = pd.DataFrame(all_cols, index=df.index)
    return list(X.columns), X


def _make_multinom_clf():
    """
    Create a multinomial logistic regression classifier compatible with
    both old sklearn (<1.5, needs multi_class='multinomial') and new
    sklearn (>=1.5, multi_class removed; lbfgs is always multinomial for >2 classes).
    """
    import sklearn
    from packaging.version import Version
    kwargs = dict(solver="lbfgs", C=1e9, max_iter=5000, tol=1e-5)
    try:
        sk_version = Version(sklearn.__version__)
        if sk_version < Version("1.5"):
            kwargs["multi_class"] = "multinomial"
    except Exception:
        pass  # if version check fails, omit multi_class and rely on default
    return LogisticRegression(**kwargs)


#def _fit_predict_multinom(train, test, feature_cols, outcome_col="outcome", return_clf=False):
def _fit_predict_multinom(train, test, feature_cols, outcome_col="outcome", return_clf=False,
                          formula_str=None):
    """Fit sklearn multinomial logistic regression and return predicted probs.

    Fixes vs original:
    - NaN rows in test are handled gracefully (NaN probs returned instead of crash)
    - max_iter increased to 5000 to match R nnet convergence behavior
    - Compatible with both sklearn <1.5 and >=1.5 (multi_class kwarg removed in 1.5)
    """
    #classes = sorted(train[outcome_col].unique())
    classes = sorted([c for c in train[outcome_col].unique() if isinstance(c, str)])
    if len(classes) < 2:
        #return None
        return (None, None) if return_clf else None 

    #X_tr = train[feature_cols].values.astype(float)
    #y_tr = train[outcome_col].values
    #X_te = test[feature_cols].values.astype(float)

    #import patsy
    #y_tr, X_tr = patsy.dmatrices(formula_str, train, return_type="dataframe")
    #X_te = patsy.dmatrix(X_tr.design_info, test, return_type="dataframe")

    # AFTER
    if formula_str is not None:
        rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
        X_tr_df = patsy.dmatrix(rhs, train, return_type="dataframe")
        X_te_df = patsy.dmatrix(X_tr_df.design_info, test, return_type="dataframe")
        X_tr_df = X_tr_df.drop(columns=["Intercept"], errors="ignore")
        X_te_df = X_te_df.drop(columns=["Intercept"], errors="ignore")
        X_te_df = X_te_df.reindex(columns=X_tr_df.columns, fill_value=0.0)  # ← fixes index OOB
        return _fit_predict_design(
            X_tr_df.values.astype(float), train[outcome_col].values, X_te_df.values.astype(float),
            list(X_tr_df.columns), test.index, return_clf=return_clf, scale=True,
        )
    return _fit_predict_design(
        train[feature_cols].values.astype(float), train[outcome_col].values,
        test[feature_cols].values.astype(float), feature_cols, test.index,
        return_clf=return_clf, scale=False,
    )


def _fit_predict_design(X_tr, y_tr, X_te, feature_cols, test_index, *, return_clf=False, scale=True):
    """The multinomial fit and its test predictions from design matrices:
    training rows are standardised (when scale) before rows with a
    non-finite feature or no outcome are dropped, as the formula path always did."""
    scaler = None
    if scale:
        from sklearn.preprocessing import StandardScaler
        # In place: both designs are this fit's own copies, and the scaled
        # values are the same as a scaled copy's.
        scaler = StandardScaler(copy=False)
        X_tr = scaler.fit_transform(X_tr)            # ← fixes convergence
        X_te = scaler.transform(X_te)
        scaler.set_params(copy=True)  # as saved before: later transforms copy

    # Drop NaN/Inf rows from training
    #bad_tr = ~np.isfinite(X_tr).all(axis=1)
    #X_tr = X_tr[~bad_tr]
    #y_tr = y_tr[~bad_tr]
    # NEW:
    bad_tr = ~np.isfinite(X_tr).all(axis=1)
    bad_outcome = np.array([not isinstance(v, str) for v in y_tr])  # ← NEW
    bad_tr = bad_tr | bad_outcome                                     # ← NEW
    X_tr = X_tr[~bad_tr]
    y_tr = y_tr[~bad_tr]
    if len(X_tr) == 0:
        #return None
        return (None, None) if return_clf else None 
    if len(np.unique(y_tr)) < 2:
        #return None
        return (None, None) if return_clf else None 

    clf = _make_multinom_clf()
    try:
        clf.fit(X_tr, y_tr)
        clf.feature_names = feature_cols
        clf.scaler_ = scaler
    except Exception as e:
        warnings.warn(f"_fit_predict_multinom fit failed: {e}")
        #return None
        return (None, None) if return_clf else None 

    # Identify NaN/Inf rows in test — return NaN probs for those
    bad_te = ~np.isfinite(X_te).all(axis=1)
    probs = np.full((len(X_te), len(clf.classes_)), np.nan)
    if (~bad_te).any():
        probs[~bad_te] = clf.predict_proba(X_te[~bad_te])

    #return pd.DataFrame(probs, columns=[f"cv_{c}" for c in clf.classes_], index=test.index)
    preds_df = pd.DataFrame(probs, columns=[f"cv_{c}" for c in clf.classes_], index=test_index)
    return (preds_df, clf) if return_clf else preds_df


def _parse_formula_cols(formula_str):
    """Extract RHS feature column names from an expanded formula string."""
    if "~" in formula_str:
        rhs = formula_str.split("~", 1)[1]
    else:
        rhs = formula_str
    cols = [t.strip() for t in re.split(r"[+\-\*:\(\)]", rhs)]
    return [c for c in cols if c and not c.isdigit() and c != "1"]


def validate_formula_feature_support(data, formulas):
    """
    Validate that every enabled formula feature exists and has finite support.

    Connect-stage rainfall-horizon metadata is used, when available, to turn
    an all-missing short-horizon predictor into a precise constructibility
    error before any model fitting starts.
    """
    feature_map = {
        model_name: list(dict.fromkeys(_parse_formula_cols(formula)))
        for model_name, formula in formulas.items()
    }
    unavailable = data.attrs.get("unavailable_rain_predictors", {})
    problems = []

    for model_name, columns in feature_map.items():
        missing = [column for column in columns if column not in data.columns]
        if missing:
            problems.append(
                f"Formula '{model_name}' is missing predictor columns: "
                f"{', '.join(missing)}"
            )

        for column in columns:
            if column not in data.columns:
                continue
            numeric = pd.to_numeric(data[column], errors="coerce").to_numpy(
                dtype=float
            )
            if np.isfinite(numeric).any():
                continue

            details = unavailable.get(column)
            if details:
                problems.append(
                    f"Formula '{model_name}': {column} cannot be constructed "
                    f"under rain_day_max={details['rain_day_max']} "
                    f"(window={details['window']}, week={details['week']}; "
                    "no complete start days)."
                )
            else:
                problems.append(
                    f"Formula '{model_name}': {column} has no finite values "
                    "and cannot be used for fitting."
                )

    if problems:
        raise ValueError(
            "Formula feature validation failed before model fitting:\n- "
            + "\n- ".join(problems)
        )
    return feature_map


def apply_formula_sample_support(data, formulas):
    """
    Apply common-complete support using only predictors in enabled formulas.

    Returns the filtered DataFrame and diagnostics. Predictors that exist in
    the connector output but are unused by all formulas do not affect support.
    """
    feature_map = validate_formula_feature_support(data, formulas)
    required = list(dict.fromkeys(
        column
        for columns in feature_map.values()
        for column in columns
    ))
    keep = np.ones(len(data), dtype=bool)
    excluded_by_column = {}
    for column in required:
        numeric = pd.to_numeric(data[column], errors="coerce").to_numpy(
            dtype=float
        )
        finite = np.isfinite(numeric)
        excluded_by_column[column] = int((~finite).sum())
        keep &= finite

    filtered = data.loc[keep].copy()
    diagnostics = {
        "policy": "common_complete",
        "required_columns": required,
        "rows_before": int(len(data)),
        "rows_after": int(len(filtered)),
        "rows_excluded": int((~keep).sum()),
        "nonfinite_by_column": {
            column: count
            for column, count in excluded_by_column.items()
            if count
        },
    }
    filtered.attrs["formula_sample_support"] = diagnostics
    return filtered, diagnostics


_GLOBAL_CV_WORKER_CONTEXT = None
_GLOBAL_CV_WORKER_THREAD_LIMIT = None


def _initialize_global_cv_worker(context, blas_threads=None):
    """Install shared global-CV inputs once in each worker process.

    ``blas_threads`` caps the worker's BLAS/OpenMP pools: each worker would
    otherwise start one thread per core, so N workers oversubscribe the
    machine N times over.
    """
    global _GLOBAL_CV_WORKER_CONTEXT, _GLOBAL_CV_WORKER_THREAD_LIMIT
    if isinstance(context, str):
        # A path to the pickled context, written once by the parent.
        with open(context, "rb") as handle:
            context = pickle.load(handle)
    _GLOBAL_CV_WORKER_CONTEXT = context
    if blas_threads:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:  # scikit-learn installs it; without it, no cap
            return
        _GLOBAL_CV_WORKER_THREAD_LIMIT = threadpool_limits(limits=int(blas_threads))


def _shared_design(formula_str, data_train, data_pred, blocks):
    """The formula's design matrices for every training and prediction row,
    in shared memory the fold workers attach to rather than copy (the blocks
    are appended to `blocks` for the caller to release); None when folds
    must build their own.

    A formula of plain numeric columns gives each row's terms from that row
    alone, so slicing these matrices by a fold's rows gives what the fold's
    own patsy design would. Formula functions (whose state could depend on
    the fold) and rows patsy would drop keep the per-fold route."""
    if _fit_columns(data_train, formula_str, []) is None or _fit_columns(data_pred, formula_str, []) is None:
        return None
    rhs = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    shared = {}
    for name, data in (("pred", data_pred), ("train", data_train)):
        if name == "train" and data_train is data_pred:
            shared["train"] = shared["pred"]
            break
        design = patsy.dmatrix(rhs, data, return_type="dataframe")
        if len(design) != len(data):
            return None
        design = design.drop(columns=["Intercept"], errors="ignore")
        columns = list(design.columns)
        if shared and columns != shared["columns"]:
            return None
        values = design.to_numpy(dtype=float)
        block = shared_memory.SharedMemory(create=True, size=max(values.nbytes, 1))
        blocks.append(block)
        # Column-major, as patsy's frame hands its values over: the scaler and
        # the solver then sum in the same order as for a fold's own design.
        np.ndarray(values.shape, values.dtype, buffer=block.buf, order="F")[...] = values
        shared[name] = (block.name, values.shape, values.dtype.str)
        shared["columns"] = columns
        del design, values
    return shared


_ATTACHED_DESIGNS = {}


def _design_rows(design, rows):
    """Rows of a shared design, column-major as patsy's design would be."""
    name, shape, dtype = design
    if name not in _ATTACHED_DESIGNS:
        _ATTACHED_DESIGNS[name] = shared_memory.SharedMemory(name=name)
    shared = np.ndarray(shape, dtype, buffer=_ATTACHED_DESIGNS[name].buf, order="F")
    out = np.empty((len(rows), shape[1]), dtype=shared.dtype, order="F")
    for column in range(shape[1]):
        out[:, column] = shared[:, column][rows]
    return out


def _compute_shared_fold(test_year, context):
    """A fold from the shared design matrices (see _shared_design)."""
    year = context["train_year"]
    train_rows = np.flatnonzero(
        np.isin(year, context["training_years"])
        & (year != test_year)
        & ~np.isin(year, context["true_holdout_years"])
    )
    test_positions = np.flatnonzero(context["pred_year"] == test_year)
    if not len(test_positions):
        raise ValueError(f"Global CV holdout year {test_year} has no test rows.")
    if not len(train_rows):
        raise ValueError(
            f"Global CV holdout year {test_year} has no training rows in "
            f"training_years={sorted(set(context['training_years']))}."
        )
    outcome = context["train_outcome"][train_rows]
    present = {value for value in pd.unique(outcome) if isinstance(value, str)}
    if context["required_classes"] is not None:
        missing = [value for value in context["required_classes"] if value not in present]
        if missing:
            raise ValueError(
                f"Global CV holdout year {test_year} training rows are missing "
                f"outcome classes: {', '.join(missing)}."
            )
    if len(present) < 2:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
    design = context["design"]
    preds, clf = _fit_predict_design(
        _design_rows(design["train"], train_rows),
        outcome,
        _design_rows(design["pred"], test_positions),
        design["columns"],
        pd.RangeIndex(len(test_positions)),
        return_clf=True,
        scale=True,
    )
    if preds is None or clf is None:
        raise ValueError(f"Global CV model fitting failed for holdout year {test_year}.")
    return test_year, test_positions, preds, _coef_artifact(test_year, clf, context)


def _coef_artifact(test_year, clf, context):
    if not context["save_coefs"]:
        return None
    rows = []
    for i, cls in enumerate(clf.classes_):
        for j, feat in enumerate(clf.feature_names):
            rows.append({
                "test_year": test_year,
                "class": cls,
                "feature": feat,
                "coefficient": float(clf.coef_[i, j]),
                "intercept": float(clf.intercept_[i]),
            })
    return {
        "coefs": pd.DataFrame(rows),
        "scaler": clf.scaler_,
        "features": clf.feature_names,
        "formula": context["formula_str"],
    }


def _compute_global_cv_fold(test_year, context=None):
    """Fit and predict one holdout year using explicit or worker context."""
    if context is None:
        context = _GLOBAL_CV_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("Global CV worker context has not been initialized.")
    if "design" in context:
        return _compute_shared_fold(test_year, context)

    data_train = context["data_train"]
    data_pred = context["data_pred"]
    training_years = context["training_years"]
    true_holdout_years = context["true_holdout_years"]
    required_classes = context["required_classes"]
    feature_cols = context["feature_cols"]
    formula_str = context["formula_str"]

    train = data_train[
        data_train["year"].isin(training_years)
        & (data_train["year"] != test_year)
        & (~data_train["year"].isin(true_holdout_years))
    ]
    test_positions = np.flatnonzero(data_pred["year"].to_numpy() == test_year)
    test = data_pred.iloc[test_positions]
    if test.empty:
        raise ValueError(f"Global CV holdout year {test_year} has no test rows.")
    if train.empty:
        raise ValueError(
            f"Global CV holdout year {test_year} has no training rows in "
            f"training_years={sorted(set(training_years))}."
        )
    if required_classes is not None:
        present_classes = {
            value for value in train["outcome"].unique()
            if isinstance(value, str)
        }
        missing_classes = [
            value for value in required_classes if value not in present_classes
        ]
        if missing_classes:
            raise ValueError(
                f"Global CV holdout year {test_year} training rows are missing "
                f"outcome classes: {', '.join(missing_classes)}."
            )

    preds, clf = _fit_predict_multinom(
        train,
        test,
        feature_cols,
        return_clf=True,
        formula_str=formula_str,
    )
    if preds is None or clf is None:
        raise ValueError(
            f"Global CV model fitting failed for holdout year {test_year}."
        )

    # The fold's rows go back as positions: the caller has every column of
    # them, so workers can hold just the columns a fit reads.
    return test_year, test_positions, preds, _coef_artifact(test_year, clf, context)


def _run_global_cv_folds(context, holdout_years, worker_count, tmp):
    if worker_count <= 1:
        return [_compute_global_cv_fold(test_year, context) for test_year in holdout_years]
    print(
        f"  Fitting {len(holdout_years)} global CV folds with "
        f"{worker_count} workers."
    )
    blas_threads = max(1, (os.cpu_count() or 1) // worker_count)
    # Each worker loads the context from one file: passed as initargs, it
    # is pickled again for every worker, one worker after another.
    context_path = os.path.join(tmp, "context.pkl")
    with open(context_path, "wb") as handle:
        pickle.dump(context, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with get_context("spawn").Pool(
        processes=worker_count,
        initializer=_initialize_global_cv_worker,
        initargs=(context_path, blas_threads),
    ) as pool:
        return list(pool.imap(_compute_global_cv_fold, holdout_years))


def _fit_columns(data, formula_str, extra):
    """The columns of ``data`` a fold's fit reads, or None when the formula
    names something other than plain columns (then the whole frame is used)."""
    features = _parse_formula_cols(formula_str)
    if not features or any(column not in data.columns for column in features):
        return None
    return list(dict.fromkeys([*features, *extra]))


def compute_cv_global(formula_str, data_train, holdout_years, training_years,
                       true_holdout_years=(), data_pred=None, n_jobs=1, save_coefs=False,
                       required_classes=None):
    """Leave-one-year-out CV using multinomial logistic, pooling all cells.

    IMPORTANT: data_train should be restrict_to_allowed(wide_df, dissemination_cells)
    and data_pred should be wide_df (all cells). Training is restricted to the
    spec-declared training_years; predictions are generated for all cells.
    """
#    print("data_pred  =  ", data_pred)
#    print("\ntrue_holdout_years  =  ", true_holdout_years)
    if data_pred is None:
        data_pred = data_train
    feature_cols = _parse_formula_cols(formula_str)
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs < 1
    ):
        raise ValueError("n_jobs must be a positive integer.")

    holdout_years = list(holdout_years)
    context = {
        "formula_str": formula_str,
        "feature_cols": feature_cols,
        "training_years": tuple(training_years),
        "true_holdout_years": tuple(true_holdout_years),
        "required_classes": (
            tuple(required_classes) if required_classes is not None else None
        ),
        "save_coefs": bool(save_coefs),
    }
    worker_count = min(n_jobs, len(holdout_years))
    blocks = []
    with tempfile.TemporaryDirectory(prefix="global-cv-") as tmp:
        # One design matrix for every fold, in shared memory each worker
        # attaches to; otherwise workers get only the columns a fit reads
        # (each holds its own copy, and full tables times the workers ran to
        # tens of GB).
        design = _shared_design(formula_str, data_train, data_pred, blocks)
        if design is not None:
            context.update(
                design=design,
                train_year=data_train["year"].to_numpy(),
                train_outcome=data_train["outcome"].to_numpy(dtype=object),
                pred_year=data_pred["year"].to_numpy(),
            )
        else:
            train_columns = _fit_columns(data_train, formula_str, ["year", "outcome"])
            pred_columns = _fit_columns(data_pred, formula_str, ["year"])
            context.update(
                data_train=data_train if train_columns is None else data_train[train_columns],
                data_pred=data_pred if pred_columns is None else data_pred[pred_columns],
            )
        try:
            fold_results = _run_global_cv_folds(context, holdout_years, worker_count, tmp)
        finally:
            for name in [b.name for b in blocks]:
                attached = _ATTACHED_DESIGNS.pop(name, None)
                if attached is not None:
                    attached.close()
            for block in blocks:
                block.close()
                block.unlink()

    if not fold_results:
        raise ValueError("Global CV produced no fold predictions.")
    cv_preds = pd.concat(
        [
            pd.concat(
                [
                    data_pred.iloc[positions].reset_index(drop=True),
                    preds.reset_index(drop=True),
                ],
                axis=1,
            )
            for _, positions, preds, _ in fold_results
        ],
        ignore_index=True,
    )
    coefs_by_year = {
        test_year: coef_artifact
        for test_year, _, _, coef_artifact in fold_results
        if coef_artifact is not None
    }
    return cv_preds, coefs_by_year


def compute_cv_local(formula_str, data, holdout_years):
    """Per-cell leave-one-year-out CV."""
    feature_cols = _parse_formula_cols(formula_str)
    unique_cells = data[["id"]].drop_duplicates()
    results = []

    for _, cell in unique_cells.iterrows():
        cell_data = data[data["id"] == cell["id"]]
        cell_holdout = [y for y in holdout_years if y in cell_data["year"].values]
        for test_year in cell_holdout:
            train = cell_data[cell_data["year"] != test_year]
            test = cell_data[cell_data["year"] == test_year]
            if train.empty or test.empty:
                continue
            #preds = _fit_predict_multinom(train, test, feature_cols)
            preds = _fit_predict_multinom(train, test, feature_cols, formula_str=formula_str)
            if preds is not None:
                result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_cv_neighbors(formula_str, data, holdout_years, lat_band=2, lon_band=2):
    """
    NOTE: Neighbor-cell CV by lat/lon distance is not applicable with adm3_name data.
    This function raises NotImplementedError.  Use compute_cv_standard instead.
    """
    raise NotImplementedError(
        "compute_cv_neighbors uses lat/lon distances and is not supported for adm3_name data. "
        "Use compute_cv_standard instead."
    )
    feature_cols = _parse_formula_cols(formula_str)
    unique_cells = data[["id"]].drop_duplicates()
    results = []

    for _, cell in unique_cells.iterrows():
        cell_data = data[data["id"] == cell["id"]]
        cell_holdout = [y for y in holdout_years if y in cell_data["year"].values]
        for test_year in cell_holdout:
            train = data[
                (data["year"] != test_year) &
                (abs(data["lat"] - cell["lat"]) <= lat_band) &
                (abs(data["lon"] - cell["lon"]) <= lon_band)
            ]
            test = data[(data["year"] == test_year) & (data["id"] == cell["id"])]
            if train.empty or test.empty:
                continue
            #preds = _fit_predict_multinom(train, test, feature_cols)
            preds = _fit_predict_multinom(train, test, feature_cols, formula_str=formula_str)
            if preds is not None:
                result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_cv_clusters(formula_str, data, holdout_years, cluster_var):
    """Cluster-based CV (train on cells in same cluster)."""
    feature_cols = _parse_formula_cols(formula_str)
    unique_cells = data[["id"]].drop_duplicates()
    results = []

    for _, cell in unique_cells.iterrows():
        cell_data = data[data["id"] == cell["id"]]
        cluster = cell_data[cluster_var].iloc[0]
        cell_holdout = [y for y in holdout_years if y in cell_data["year"].values]
        for test_year in cell_holdout:
            train = data[(data["year"] != test_year) & (data[cluster_var] == cluster)]
            test = data[(data["year"] == test_year) & (data["id"] == cell["id"])]
            if train.empty or test.empty:
                continue
            #preds = _fit_predict_multinom(train, test, feature_cols)
            preds = _fit_predict_multinom(train, test, feature_cols, formula_str=formula_str)
            if preds is not None:
                result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def restrict_to_allowed(df, allowed_cells):
    """Filter df to rows whose id is in allowed_cells."""
    if allowed_cells is None:
        return df
    return df[df["id"].isin(set(allowed_cells["id"]))]


def clean_probs5(P):
    """Sanitize 5-bin probability matrix: replace non-finite with 0, normalize rows."""
    P = np.array(P, dtype=float)
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= 0).flatten()
    if zero.any():
        P[zero] = 1.0 / P.shape[1]
        rs = P.sum(axis=1, keepdims=True)
    return P / rs


def pooled_rps5(P, y_onehot):
    """Pooled RPS from 5-bin probability and one-hot matrices."""
    CP = np.cumsum(P, axis=1)
    CY = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum((CP - CY) ** 2, axis=1)))


def _fast_auc(y01, score):
    """Efficient AUC via Wilcoxon rank-sum (matches R's fast_auc)."""
    ok = np.isfinite(score) & ~np.isnan(y01)
    y = y01[ok].astype(int)
    s = score[ok]
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    # The same average ranks as pd.Series(s).rank(method="average"), without
    # building a Series per call (this runs once per cell per metric).
    ranks = rankdata(s, method="average")
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)


def compute_fair_brier5(cv_preds, allowed_cells, n_train=30):
    """Compute fair (finite-sample corrected) Brier score over the interval bins
    (week1..weekN + later), inferred from the cv_week* columns present."""
    bins = [f"week{w}" for w in _cv_weeks(cv_preds)] + ["later"]
    cols = [f"cv_{b}" for b in bins]
    sub = restrict_to_allowed(cv_preds, allowed_cells)
    sub = sub.dropna(subset=["outcome"] + cols)

    P = sub[cols].values
    Y = np.column_stack([(sub["outcome"] == b).astype(float) for b in bins])
    bs_rows = np.sum((P - Y) ** 2, axis=1)
    corr_rows = np.sum(P * (1 - P), axis=1) / (n_train - 1)
    return float(np.mean(bs_rows - corr_rows))


def compute_cell_metrics_fast(df, allowed_cells=None):
    """
    Compute per-cell and pooled Brier/RPS/AUC/pietra from cv_* probability columns.

    Matches R compute_cell_metrics_fast:
    - Brier uses the interval bins week1..weekN + later (NOT 'earlier'),
      inferred from the cv_week* columns present
    - RPS prepends 'earlier' when a cv_earlier column is present
    - Pooled AUC uses the fast Wilcoxon rank-sum method
    - Per-bin Brier and AUC are computed for the 'ALL' row
    - Pietra (KS statistic between pos/neg score distributions) is included
    - n (number of observations) is included
    - Returns per-cell rows PLUS an 'ALL' pooled row
    """
    # Interval bins (week1..weekN + later), inferred from cv_week* columns
    _wk = [f"week{w}" for w in _cv_weeks(df)]
    bins5 = _wk + ["later"]
    cv_cols5 = [f"cv_{b}" for b in bins5]

    # RPS bins: prepend 'earlier' if that column is present
    bins_rps_desired = (["earlier"] if "cv_earlier" in df.columns else []) + bins5
    present_rps = [b for b in bins_rps_desired if f"cv_{b}" in df.columns]
    cv_cols_rps = [f"cv_{b}" for b in present_rps]

    sub = df.dropna(subset=["outcome"] + cv_cols5).copy()
    if allowed_cells is not None:
        sub = sub[sub["id"].isin(set(allowed_cells["id"]))]
    sub = sub.reset_index(drop=True) # --NEW

    if sub.empty:
        return pd.DataFrame()

    P5 = sub[cv_cols5].values.astype(float)
    Y5 = np.column_stack([(sub["outcome"] == b).astype(float) for b in bins5])

    # RPS matrix (6-bin if earlier available)
    if cv_cols_rps != cv_cols5:
        P_rps = np.zeros((len(sub), len(present_rps)), dtype=float)
        for j, b in enumerate(present_rps):
            col = f"cv_{b}"
            if col in sub.columns:
                vals = sub[col].values.astype(float)
                vals[~np.isfinite(vals)] = 0.0
                P_rps[:, j] = vals
        Y_rps = np.column_stack([(sub["outcome"] == b).astype(float) for b in present_rps])
    else:
        P_rps = P5.copy()
        Y_rps = Y5.copy()

    # --- Per-cell metrics ---
    rows = []
    grouped = sub.groupby("id")
    positions = grouped.indices  # the rows' positions; sub's index is 0..n-1
    for cell_id, g_idx in grouped.groups.items():
        #g_P5 = P5[sub.index.get_indexer(g_idx)]
        #g_Y5 = Y5[sub.index.get_indexer(g_idx)]
        #g_Prps = P_rps[sub.index.get_indexer(g_idx)]
        #g_Yrps = Y_rps[sub.index.get_indexer(g_idx)]
        pos = positions[cell_id]
        g_P5 = P5[pos]
        g_Y5 = Y5[pos]
        g_Prps = P_rps[pos]
        g_Yrps = Y_rps[pos]

        brier = float(np.mean(np.sum((g_P5 - g_Y5) ** 2, axis=1)))
        rps = pooled_rps5(g_Prps, g_Yrps)
        auc = _fast_auc(g_Y5.flatten().astype(int), g_P5.flatten())
        rows.append({"id": cell_id, "brier": brier, "rps": rps, "auc": auc, "n": len(g_idx)})

    cell_metrics = pd.DataFrame(rows)

    # --- Pooled 'ALL' row ---
    # Need positional indexing since sub may have non-contiguous index
    sub_reset = sub.reset_index(drop=True)
    P5_all = sub_reset[cv_cols5].values.astype(float)
    Y5_all = np.column_stack([(sub_reset["outcome"] == b).astype(float) for b in bins5])
    P_rps_all = sub_reset[cv_cols_rps].values.astype(float) if cv_cols_rps != cv_cols5 else P5_all.copy()
    Y_rps_all = (np.column_stack([(sub_reset["outcome"] == b).astype(float) for b in present_rps])
                 if cv_cols_rps != cv_cols5 else Y5_all.copy())
    for j in range(P_rps_all.shape[1]):
        P_rps_all[~np.isfinite(P_rps_all[:, j]), j] = 0.0

    brier_all = float(np.sum((P5_all - Y5_all) ** 2) / len(P5_all))
    rps_all = pooled_rps5(P_rps_all, Y_rps_all)
    auc_all = _fast_auc(Y5_all.flatten().astype(int), P5_all.flatten())

    # Per-bin Brier and AUC (5 bins, matches R)
    brier_by_bin = {}
    auc_by_bin = {}
    for j, b in enumerate(bins5):
        brier_by_bin[f"brier_{b}"] = float(np.mean((P5_all[:, j] - Y5_all[:, j]) ** 2))
        auc_by_bin[f"auc_{b}"] = _fast_auc(Y5_all[:, j].astype(int), P5_all[:, j])

    # Pietra: KS statistic between positive and negative score distributions
    p_flat = P5_all.flatten()
    y_flat = Y5_all.flatten().astype(int)
    pos_scores = p_flat[y_flat == 1]
    neg_scores = p_flat[y_flat == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        pietra = float(ks_2samp(pos_scores, neg_scores).statistic)
    else:
        pietra = np.nan

    pooled_row = {
        "id": "ALL",
        "lat": np.nan,
        "lon": np.nan,
        "brier": brier_all,
        "rps": rps_all,
        "n": len(sub_reset),
        "auc": auc_all,
        "pietra": pietra,
    }
    pooled_row.update(brier_by_bin)
    pooled_row.update(auc_by_bin)

    pooled_df = pd.DataFrame([pooled_row])
    return pd.concat([cell_metrics, pooled_df], ignore_index=True)


def compute_yearly_metrics(cv_preds, model_name, method, n_bins):
    """Per-year pooled Brier/RPS/AUC rows over the interval bins."""
    bins = interval_bins(n_bins)
    cv_cols = [f"cv_{b}" for b in bins]
    sub = cv_preds.dropna(subset=["outcome"] + cv_cols)
    rows = []
    for yr, g in sub.groupby("year"):
        P = g[cv_cols].values.astype(float)
        Y = np.column_stack([(g["outcome"] == b).astype(float) for b in bins])
        brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
        rps = pooled_rps5(P, Y)
        y_long = Y.flatten()
        p_long = P.flatten()
        try:
            auc = roc_auc_score(y_long, p_long) if len(np.unique(y_long)) == 2 else np.nan
        except Exception:
            auc = np.nan
        rows.append({
            "year": yr, "model": model_name, "cv_method": method,
            "brier": brier, "rps": rps, "auc": auc,
        })
    return rows


def score_model_predictions(cv_preds, model_name, method, allowed_cells, n_bins):
    """Score CV predictions as one model: (per-cell metrics frame, yearly metric rows)."""
    cell_metrics = compute_cell_metrics_fast(cv_preds, allowed_cells=allowed_cells)
    cell_metrics["model"] = model_name
    cell_metrics["cv_method"] = method
    return cell_metrics, compute_yearly_metrics(cv_preds, model_name, method, n_bins)


# ---------------------------------------------------------------------------
# Externally produced CV predictions (extras.external_predictions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExternalPredictions:
    """A spec entry naming a pickled cv_preds frame produced outside this script."""
    name: str
    path: str
    method: str = "global"


def parse_external_predictions(extras, results_dir):
    """Parse extras.external_predictions; relative files resolve against results_dir."""
    entries = (extras or {}).get("external_predictions") or []
    return [_parse_external_entry(entry, results_dir) for entry in entries]


def _parse_external_entry(entry, results_dir):
    missing = [key for key in ("name", "file") if not (entry or {}).get(key)]
    if missing:
        raise ValueError(
            f"extras.external_predictions entry {entry!r} is missing: {', '.join(missing)}"
        )
    path = os.path.join(results_dir, entry["file"])
    return ExternalPredictions(
        name=str(entry["name"]),
        path=path,
        method=str(entry.get("method") or "global"),
    )


def load_external_predictions(external, n_bins):
    """Load an external cv_preds pickle, requiring the columns scoring reads."""
    with open(external.path, "rb") as handle:
        cv_preds = pickle.load(handle)
    if not isinstance(cv_preds, pd.DataFrame):
        raise ValueError(
            f"External predictions '{external.name}' ({external.path}) must be a "
            f"pandas DataFrame, got {type(cv_preds).__name__}."
        )
    _require_scoring_columns(cv_preds, external, n_bins)
    if cv_preds.empty:
        raise ValueError(
            f"External predictions '{external.name}' ({external.path}) has no rows."
        )
    return cv_preds.reset_index(drop=True)


def _require_scoring_columns(cv_preds, external, n_bins):
    expected_cv = [f"cv_{b}" for b in interval_bins(n_bins)]
    required = ["id", "year", "outcome"] + expected_cv
    missing = [c for c in required if c not in cv_preds.columns]
    unexpected = [f"cv_week{w}" for w in _cv_weeks(cv_preds) if w > n_bins]
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing columns {missing}")
        if unexpected:
            problems.append(f"unexpected columns {unexpected} beyond {n_bins} week bins")
        raise ValueError(
            f"External predictions '{external.name}' ({external.path}): "
            f"{'; '.join(problems)}. Expected {required}."
        )


def summarize_models_pooled(all_cells, baseline_model="unc_clim_raw"):
    """Compute skill scores vs a baseline from the 'ALL' row."""
    model_avg = all_cells[all_cells["id"] == "ALL"].copy()
    clim_rows = model_avg[model_avg["model"] == baseline_model]
    if clim_rows.empty:
        raise ValueError(f"Baseline model '{baseline_model}' not found in all_cells.")
    clim_row = clim_rows.iloc[0]
    brier_clim = clim_row["brier"]
    rps_clim = clim_row["rps"]
    auc_clim = clim_row["auc"]
    model_avg["brier_skill"] = 1.0 - (model_avg["brier"] / brier_clim)
    model_avg["rps_skill"] = 1.0 - (model_avg["rps"] / rps_clim)
    model_avg["AUC diff"] = (model_avg["auc"] - auc_clim) * 100.0
    return model_avg


def summarize_maps_compare(all_cells, method, clim_model="clim_raw", final_model="blended_model"):
    """Per-cell comparison of a final model vs climatology."""
    clim_df = (all_cells[all_cells["model"] == clim_model]
               .rename(columns={"brier": "brier_clim", "rps": "rps_clim", "auc": "auc_clim"})
               [["id", "brier_clim", "rps_clim", "auc_clim"]])
    final_df = (all_cells[all_cells["model"] == final_model]
                .rename(columns={"brier": "brier_final", "rps": "rps_final", "auc": "auc_final"})
                [["id", "brier_final", "rps_final", "auc_final"]])
    merged = clim_df.merge(final_df, on="id", how="outer")
    merged["auc_diff"] = merged["auc_final"] - merged["auc_clim"]
    merged["brier_skill"] = 1.0 - (merged["brier_final"] / merged["brier_clim"])
    merged["rps_skill"] = 1.0 - (merged["rps_final"] / merged["rps_clim"])
    merged["cv_method"] = method
    return merged


# ---------------------------------------------------------------------------
# MME blend optimization (Section 4 — was missing from Python translation)
# ---------------------------------------------------------------------------

def simplex_grid(n, step=0.1):
    """Generate all weight vectors on the N-simplex with given step size."""
    if n == 1:
        return np.array([[1.0]])
    vals = np.arange(0, 1 + step / 2, step)
    from itertools import product as iproduct
    grid = []
    for combo in iproduct(*([vals] * (n - 1))):
        s = sum(combo)
        if s <= 1.0 + 1e-9:
            last = 1.0 - s
            grid.append(list(combo) + [last])
    return np.array(grid)


def softmax_n(theta):
    """Softmax for N dimensions."""
    e = np.exp(theta - np.max(theta))
    return e / e.sum()


def optimize_mme_weights(P_allowed_clean, Y_allowed, w_col_names, mc_cores=1):
    """
    Optimize MME blend weights to minimize pooled RPS.

    Uses coarse simplex grid search (step=0.1) to initialize BFGS,
    matching R's Section 4 strategy exactly.

    Parameters
    ----------
    P_allowed_clean : list of np.ndarray
        List of (n_obs x 5) cleaned probability matrices, one per blend source.
    Y_allowed : np.ndarray
        (n_obs x 5) one-hot outcome matrix restricted to allowed cells.
    w_col_names : list of str
        Names for the weight columns (e.g. ["w_clim_raw", "w_ngcm_calibrated"]).
    mc_cores : int
        Number of parallel cores (not used here; kept for signature parity with R).

    Returns
    -------
    dict with keys:
        "weights": np.ndarray of optimal weights
        "rps": float optimal RPS
        "weights_df": pd.DataFrame with objective/weight columns
    """
    from scipy.optimize import minimize

    n_blend = len(P_allowed_clean)

    def fast_rps_from_w(w):
        P_mix = sum(wi * Pi for wi, Pi in zip(w, P_allowed_clean))
        P_mix = clean_probs5(P_mix)
        return pooled_rps5(P_mix, Y_allowed)

    # Coarse grid search
    weight_mat = simplex_grid(n_blend, step=0.1)
    best_rps = np.inf
    best_w = weight_mat[0]
    for w in weight_mat:
        rps = fast_rps_from_w(w)
        if np.isfinite(rps) and rps < best_rps:
            best_rps = rps
            best_w = w

    # BFGS refinement in unconstrained softmax space
    w0 = np.maximum(best_w, 1e-6)
    w0 = w0 / w0.sum()
    theta0 = np.log(w0)

    result = minimize(
        fun=lambda theta: fast_rps_from_w(softmax_n(theta)),
        x0=theta0,
        method="BFGS",
        options={"maxiter": 200},
    )
    w_star = softmax_n(result.x)
    opt_rps = fast_rps_from_w(w_star)

    weights_df = pd.DataFrame([{"objective": "rps", **dict(zip(w_col_names, w_star)), "rps": opt_rps}])
    return {"weights": w_star, "rps": opt_rps, "weights_df": weights_df}


def apply_mme_weights(mme_sources, blend_names, opt_w, id_vars, bins5=None):
    """
    Apply optimized MME weights to produce blended cv_* predictions.

    Parameters
    ----------
    mme_sources : dict name -> DataFrame with cv_* columns
    blend_names : list of str
    opt_w : np.ndarray of weights (same order as blend_names)
    id_vars : list of str (columns to keep as metadata)
    bins5 : list of str, defaults to the interval bins inferred from the first
        source's cv_week* columns (week1..weekN + later).

    Returns
    -------
    DataFrame with id_vars + cv_week1..cv_later
    """
    if bins5 is None:
        first = mme_sources[blend_names[0]]
        bins5 = [f"week{w}" for w in _cv_weeks(first)] + ["later"]
    cols5 = [f"cv_{b}" for b in bins5]

    # Join all sources on id_vars
    join_list = []
    for nm in blend_names:
        src = mme_sources[nm].copy()
        rename_map = {c: f"cv_{nm}_{c[3:]}" for c in cols5 if c in src.columns}
        src = src[id_vars + [c for c in cols5 if c in src.columns]].rename(columns=rename_map)
        join_list.append(src)

    base = join_list[0]
    for j in range(1, len(join_list)):
        base = base.merge(join_list[j], on=id_vars, how="inner")

    # Build P matrices and weighted sum
    P_list = []
    for nm in blend_names:
        sel_cols = [f"cv_{nm}_{b}" for b in bins5]
        P_list.append(base[sel_cols].values.astype(float))

    P_mix = sum(wi * Pi for wi, Pi in zip(opt_w, P_list))
    P_mix = clean_probs5(P_mix)

    result = base[id_vars].copy()
    for j, b in enumerate(bins5):
        result[f"cv_{b}"] = P_mix[:, j]
    return result


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def exclude_for_mapping(df):
    """Exclude NE and SW island cells."""
    return df[~((df["lon"] > 90) | ((df["lon"] < 75) & (df["lat"] < 12)))]


def create_cell_polygon(lat, lon, half_size=1):
    """Create a rectangular polygon for a grid cell center."""
    return pd.DataFrame({
        "lon": [lon - half_size, lon + half_size, lon + half_size, lon - half_size, lon - half_size],
        "lat": [lat - half_size, lat - half_size, lat + half_size, lat + half_size, lat - half_size],
    })


def build_polygons_for_mapping(grid_centers, allowed_cells, half_size=1):
    """Build cell polygons for choropleth mapping."""
    poly_data = grid_centers[["id"]].drop_duplicates()
    poly_data = exclude_for_mapping(poly_data)

    polys = []
    for _, cell in poly_data.iterrows():
        poly = create_cell_polygon(cell["lat"], cell["lon"], half_size=half_size)
        poly["id"] = cell["id"]
        poly["lat_center"] = cell["lat"]
        poly["lon_center"] = cell["lon"]
        polys.append(poly)

    polygons_df = pd.concat(polys, ignore_index=True) if polys else pd.DataFrame()
    allowed_ids = set(allowed_cells["id"])
    allowed_polygons_df = polygons_df[polygons_df["id"].isin(allowed_ids)] if not polygons_df.empty else polygons_df
    return {"polygons_df": polygons_df, "allowed_polygons_df": allowed_polygons_df}
