"""evaluate() + Evaluation.save() write exactly what 1_blend_evaluation.py wrote."""

import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from onset_blending._legacy.blend_evaluation_utils import (
    _present_weeks,
    apply_formula_sample_support,
    build_formulas_from_spec,
    clean_probs5,
    compute_cell_metrics_fast,
    compute_cv_clusters,
    compute_cv_global,
    compute_cv_local,
    configured_holdout_years,
    fit_platt_weights_export,
    forecast_label,
    get_forecast_variant_suffix,
    load_external_predictions,
    make_calibrated_preds_from_wide,
    make_cutoff_tag,
    make_raw_preds_from_wide,
    make_raw_preds_from_wide_logit,
    make_raw_preds_from_wide_logit_window,
    make_year_tag,
    optimize_mme_weights,
    parse_external_predictions,
    pooled_rps5,
    resolve_climatology_weeks,
    restrict_to_allowed,
    score_model_predictions,
    summarize_models_pooled,
)
from onset_blending._legacy.misc import coalesce, interval_bins
from onset_blending._legacy.nc_utils import ensure_id_col
from onset_blending._legacy.spatial_id_utils import (
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from onset_blending.evaluation import evaluate

BINS = interval_bins(2)


def _script(spec, wide_df, dissemination_input, path_box, cores=1):
    """1_blend_evaluation.py's main() as it was, after reading its inputs.

    Prints, diagnostic comments and the warn-and-skip branches these tests do
    not reach are left out; the rest is unchanged."""
    cutoff_mode = coalesce(spec.get("run", {}).get("cutoff_mode"), "ref_onset")
    training_years = list(
        map(int, coalesce(spec.get("run", {}).get("training_years"), list(range(2000, 2025))))
    )
    true_holdout_years = list(map(int, coalesce(spec.get("run", {}).get("true_holdout_years"), [])))
    holdout_years = configured_holdout_years(spec)
    cv_methods = coalesce(spec.get("cv", {}).get("methods"), ["global"])
    cutoff_tag = make_cutoff_tag(cutoff_mode)
    year_tag = make_year_tag(holdout_years)
    output_tag = f"{cutoff_tag}{year_tag}"

    wide_df = ensure_id_col(wide_df, spec=spec, context="blend input IDs")
    id_convention = resolve_grid_id_convention(
        spec=spec, authoritative_ids=wide_df["id"], context="blend input/dissemination ID handoff"
    )
    cell_cfg = spec["cell"]
    dissemination_id_col = cell_cfg.get("id_col") or cell_cfg.get("dissemination_id_col")
    if dissemination_id_col:
        if dissemination_id_col not in dissemination_input.columns:
            raise ValueError(
                f"cell.id_col '{dissemination_id_col}' not found. "
                f"Found: {dissemination_input.columns.tolist()}"
            )
        dissemination_input = dissemination_input.rename(columns={dissemination_id_col: "id"})
    dissemination_cells = ensure_id_col(
        dissemination_input, spec=spec, convention=id_convention, context="blend dissemination IDs"
    )
    validate_id_coordinate_consistency(dissemination_cells, context="blend dissemination IDs")
    dissemination_cells = dissemination_cells.drop_duplicates("id")

    _, clim_weeks = resolve_climatology_weeks(spec, wide_df)
    N_BINS = max(clim_weeks)
    external_predictions = [
        (external, load_external_predictions(external, N_BINS))
        for external in parse_external_predictions(spec.get("extras"), path_box)
    ]

    formulas, formula_resolution = build_formulas_from_spec(
        spec, cutoff_mode, data=wide_df, return_resolution=True, n_bins=N_BINS
    )
    wide_df, support = apply_formula_sample_support(wide_df, formulas)
    support["formula_resolution"] = formula_resolution
    with open(os.path.join(path_box, f"formula_sample_support{output_tag}.json"), "w") as handle:
        json.dump(support, handle, indent=2)

    wide_df_allowed = restrict_to_allowed(wide_df, dissemination_cells)
    all_cells_list = []
    yearly_metrics_list = []
    clim_cv_preds_list = {}
    cal_preds_store = {}

    for model_name, formula_str in formulas.items():
        for method in cv_methods:
            try:
                if method == "global":
                    cv_preds, coefs_by_year = compute_cv_global(
                        formula_str,
                        wide_df_allowed,
                        holdout_years,
                        training_years=training_years,
                        true_holdout_years=true_holdout_years,
                        data_pred=wide_df,
                        n_jobs=cores or 1,
                        save_coefs=True,
                        required_classes=interval_bins(N_BINS),
                    )
                    cv_preds = cv_preds.reset_index(drop=True)
                elif method == "local":
                    cv_preds = compute_cv_local(formula_str, wide_df, holdout_years)
                    coefs_by_year = {}
                elif method.startswith("cluster"):
                    cluster_var = spec.get("cv", {}).get("cluster_var", "cluster")
                    cv_preds = compute_cv_clusters(formula_str, wide_df, holdout_years, cluster_var)
                    coefs_by_year = {}
                else:
                    warnings.warn(f"Unknown CV method: {method}. Skipping.")
                    continue
                if cv_preds is None or cv_preds.empty:
                    raise ValueError(f"No predictions for {model_name}/{method}.")
                cell_metrics, yearly_rows = score_model_predictions(
                    cv_preds, model_name, method, dissemination_cells, N_BINS
                )
                all_cells_list.append(cell_metrics)
                yearly_metrics_list.extend(yearly_rows)
                with open(
                    os.path.join(path_box, f"cv_preds_{model_name}_{method}{output_tag}.pkl"), "wb"
                ) as f:
                    pickle.dump(cv_preds, f)
                if method == "global" and coefs_by_year:
                    for yr, coef_df in coefs_by_year.items():
                        coef_path = os.path.join(
                            path_box, f"coefs_{model_name}_{method}{output_tag}_year{yr}.pkl"
                        )
                        with open(coef_path, "wb") as f:
                            pickle.dump(coefs_by_year[yr], f)
            except Exception as e:
                raise RuntimeError(f"Error in {model_name}/{method}: {e}") from e

    for external, cv_preds in external_predictions:
        cell_metrics, yearly_rows = score_model_predictions(
            cv_preds, external.name, external.method, dissemination_cells, N_BINS
        )
        all_cells_list.append(cell_metrics)
        yearly_metrics_list.extend(yearly_rows)

    extras = spec.get("extras") or {}
    for task in extras.get("clim_logits") or []:
        nm = task["name"]
        base_prefix = task["base_col_prefix"]
        win_starts = [int(y) for y in (task.get("window_start_years") or [])]
        win_end = task.get("window_end_year")

        def handle_one_clim(model_nm, start_year=None, end_year=None):
            try:
                if model_nm == "unc_clim_raw":
                    cvp = make_raw_preds_from_wide_logit(
                        wide_df=wide_df,
                        base_col_prefix=base_prefix,
                        holdout_years=holdout_years,
                        earlier_col=task.get("earlier_col"),
                        earlier_is_logit=bool(task.get("earlier_is_logit", True)),
                        add_cv_earlier=True,
                        renormalize_6=True,
                    )
                else:
                    cvp = make_raw_preds_from_wide_logit_window(
                        wide_df=wide_df,
                        base_col_prefix=base_prefix,
                        holdout_years=holdout_years,
                        start_year=start_year,
                        end_year=end_year,
                    )
                if cvp is None or cvp.empty:
                    return
                clim_cv_preds_list[model_nm] = cvp
                cell_metrics = compute_cell_metrics_fast(cvp, allowed_cells=dissemination_cells)
                cell_metrics["model"] = model_nm
                cell_metrics["cv_method"] = "raw"
                all_cells_list.append(cell_metrics)
            except Exception as e:
                warnings.warn(f"Error in clim logit '{model_nm}': {e}")

        if not win_starts or win_end is None or nm != "clim_raw":
            handle_one_clim(nm, start_year=None, end_year=None)
        else:
            for sy in win_starts:
                model_nm = nm if sy == 1900 else f"{nm}_{sy}_{win_end}"
                handle_one_clim(model_nm, start_year=sy, end_year=win_end)

    for fc_cfg in extras.get("forecasts") or []:
        fc_name = fc_cfg["name"]
        variant = coalesce(fc_cfg.get("variant"), "base")
        base_label = forecast_label(fc_name, variant)
        if fc_cfg.get("raw", False):
            preds = make_raw_preds_from_wide(wide_df, fc_name, variant, holdout_years, spec)
            if preds is not None and not preds.empty:
                cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                cell_metrics["model"] = f"{base_label}_raw"
                cell_metrics["cv_method"] = "raw"
                all_cells_list.append(cell_metrics)
        if fc_cfg.get("calibrated", False):
            preds = make_calibrated_preds_from_wide(
                wide_df,
                fc_name,
                variant,
                training_years,
                holdout_years,
                true_holdout_years,
                dissemination_cells,
                spec,
            )
            if preds is not None and not preds.empty:
                key = fc_name if variant == "base" else f"{fc_name}_calibrated_{variant}"
                cal_preds_store[key] = preds
                model_nm = (
                    f"{fc_name}_calibrated"
                    if variant == "base"
                    else f"{fc_name}_calibrated_{variant}"
                )
                cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                cell_metrics["model"] = model_nm
                cell_metrics["cv_method"] = "calibrated"
                all_cells_list.append(cell_metrics)
                if fc_cfg.get("export_platt_weights", False):
                    base = f"{fc_name}_p_onset{get_forecast_variant_suffix(spec, variant)}"
                    weeks = _present_weeks(wide_df, base)
                    if weeks:
                        platt_df_full = wide_df[
                            wide_df["year"].isin(training_years + holdout_years)
                        ].copy()
                        wk_labels = [f"week{w}" for w in weeks]
                        for w in weeks:
                            platt_df_full[f"week{w}"] = platt_df_full[f"{base}_week{w}"]
                        platt_df_full["later"] = np.maximum(
                            0.0, 1.0 - platt_df_full[wk_labels].sum(axis=1)
                        )
                        platt_result = fit_platt_weights_export(
                            platt_df_full, wk_labels + ["later"], training_years, year_col="year"
                        )
                        platt_path = os.path.join(
                            path_box, f"platt_weights_{base_label}_calibrated_df{output_tag}.pkl"
                        )
                        with open(platt_path, "wb") as fp:
                            pickle.dump(platt_result["weights_df"], fp)

    mme_cfg = spec.get("mme") or {}
    if mme_cfg.get("enabled", False) and clim_cv_preds_list:
        mme_variants = list(mme_cfg.get("variants") or ["fixed_cutoff"])
        blend_models = mme_cfg.get("blend_models") or []
        bins5 = interval_bins(N_BINS)
        cols5 = [f"cv_{b}" for b in bins5]
        id_vars = ["time", "id", "lat", "lon", "year", "outcome"]
        n_blend = len(blend_models)
        blend_names = [bm["name"] for bm in blend_models]
        w_col_names = [f"w_{nm}" for nm in blend_names]
        do_opt_this_run = bool(
            mme_cfg.get("optimize_if_holdout_is_full_2000_2024", True)
        ) and sorted(holdout_years) == list(range(2000, 2025))
        for mme_variant in mme_variants:
            mme_sources = {}
            for bm in blend_models:
                if bm.get("source") == "clim":
                    src = clim_cv_preds_list.get(bm["name"])
                else:
                    cal_variant = bm.get("cal_variant", "base")
                    key = (
                        bm["name"]
                        if cal_variant == "base"
                        else f"{bm['name']}_calibrated_{cal_variant}"
                    )
                    src = cal_preds_store.get(key)
                mme_sources[bm["name"]] = src
            join_list = []
            for nm in blend_names:
                src = mme_sources[nm].copy()
                rename_map = {c: f"cv_{nm}_{c[3:]}" for c in cols5 if c in src.columns}
                keep = [v for v in id_vars if v in src.columns] + [
                    c for c in cols5 if c in src.columns
                ]
                join_list.append(src[keep].rename(columns=rename_map))
            mme_base_full = join_list[0]
            for j in range(1, n_blend):
                mme_base_full = mme_base_full.merge(join_list[j], on=id_vars, how="inner")
            mme_base_allowed = restrict_to_allowed(mme_base_full, dissemination_cells)
            P_full_list = []
            P_allowed_list = []
            for nm in blend_names:
                sel_cols = [f"cv_{nm}_{b}" for b in bins5]
                P_full_list.append(mme_base_full[sel_cols].values.astype(float))
                P_allowed_list.append(mme_base_allowed[sel_cols].values.astype(float))
            Y_allowed = np.column_stack(
                [(mme_base_allowed["outcome"] == b).astype(float) for b in bins5]
            )
            P_allowed_clean = [clean_probs5(P) for P in P_allowed_list]
            clim_model_name = next(
                (bm["name"] for bm in blend_models if bm.get("source") == "clim"), blend_names[0]
            )
            mme_prefix = f"mme_{mme_variant}_{clim_model_name}_opt"
            mme_opt_file_write = os.path.join(
                path_box,
                f"mme_optimized_weights{cutoff_tag}_{clim_model_name}_{mme_variant}{year_tag}.pkl",
            )
            weights_read_tag = mme_cfg.get("weights_year_tag") or year_tag
            mme_opt_file_read = os.path.join(
                path_box,
                f"mme_optimized_weights{cutoff_tag}_{clim_model_name}_{mme_variant}{weights_read_tag}.pkl",
            )
            if do_opt_this_run:
                opt_result = optimize_mme_weights(P_allowed_clean, Y_allowed, w_col_names, cores)
                opt_w = opt_result["weights"]
                with open(mme_opt_file_write, "wb") as fp:
                    pickle.dump(opt_result["weights_df"], fp)
            else:
                with open(mme_opt_file_read, "rb") as fp:
                    saved_weights_df = pickle.load(fp)
                row = saved_weights_df[saved_weights_df["objective"] == "rps"].iloc[0]
                opt_w = np.array([row[w] for w in w_col_names])
                P_mix = clean_probs5(sum(wi * Pi for wi, Pi in zip(opt_w, P_allowed_clean)))
                pooled_rps5(P_mix, Y_allowed)
            P_mix_full = sum(wi * clean_probs5(Pi) for wi, Pi in zip(opt_w, P_full_list))
            P_mix_full = clean_probs5(P_mix_full)
            cv_preds_mme = mme_base_full[[v for v in id_vars if v in mme_base_full.columns]].copy()
            for j, b in enumerate(bins5):
                cv_preds_mme[f"cv_{b}"] = P_mix_full[:, j]
            cell_metrics = compute_cell_metrics_fast(
                cv_preds_mme, allowed_cells=dissemination_cells
            )
            cell_metrics["model"] = f"{mme_prefix}_rps"
            cell_metrics["cv_method"] = "mme"
            all_cells_list.append(cell_metrics)

    if all_cells_list:
        all_cells = pd.concat(all_cells_list, ignore_index=True)
        summary_path = os.path.join(path_box, f"summary_models{output_tag}.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(all_cells, f)
        all_cells.to_csv(summary_path.replace(".pkl", ".csv"), index=False)
        pooled = summarize_models_pooled(all_cells)
        pooled.to_csv(os.path.join(path_box, f"summary_models_pooled{output_tag}.csv"), index=False)
    if yearly_metrics_list:
        yearly_df = pd.DataFrame(yearly_metrics_list)
        yearly_path = os.path.join(path_box, f"yearly_metrics_global{output_tag}.pkl")
        with open(yearly_path, "wb") as f:
            pickle.dump(yearly_df, f)
        yearly_df.to_csv(yearly_path.replace(".pkl", ".csv"), index=False)


def weekly_table(years, seed=0, n_cells=5, issues=4):
    rng = np.random.default_rng(seed)
    rows = [
        {
            "id": f"unit{c}",
            "time": f"{year}-06-{1 + 7 * k:02d}",
            "year": year,
            "lat": 10.0 + c,
            "lon": 40.0 + c,
            "cluster": "AB"[c % 2],
        }
        for c in range(n_cells)
        for year in years
        for k in range(issues)
    ]
    frame = pd.DataFrame(rows)
    n = len(frame)
    frame["outcome"] = rng.choice(BINS, size=n)
    frame.loc[rng.random(n) < 0.05, "outcome"] = np.nan
    frame["prob_clim_mr_unc_earlier"] = rng.normal(-2.0, 0.5, n)
    for w in (1, 2):
        for prefix in ("prob_clim_mr", "prob_clim_mr_unc", "prob_clim_mr_2018_2024"):
            frame[f"{prefix}_week{w}"] = rng.normal(-1.0, 0.5, n)
        frame[f"fc_p_onset_fixed_cutoff_week{w}"] = rng.uniform(0.0, 0.5, n)
        frame[f"diff_fc_week{w}"] = rng.normal(size=n)
    frame.loc[3, "diff_fc_week1"] = np.nan  # outside the formula sample support
    return frame


def cv_spec(training_years, cv_holdout_years, true_holdout_years, methods):
    return {
        "run": {
            "cutoff_mode": "fixed_cutoff",
            "training_years": list(training_years),
            "cv_holdout_years": list(cv_holdout_years),
            "true_holdout_years": list(true_holdout_years),
        },
        "cv": {"methods": list(methods)},
        "cell": {"id_col": "unit"},
        "models": {
            "formulas": {
                "blend": {"text": "outcome ~ prob_clim_mr_qx * diff_fc_qx"},
                "clim_only": {"text": "outcome ~ prob_clim_mr_qx"},
                "off": {"enabled": False, "text": "outcome ~ absent_qx"},
            },
            "window_variants": {"enabled": False},
        },
        "mme": {
            "enabled": True,
            "variants": ["fixed_cutoff"],
            "blend_models": [{"name": "clim_raw", "source": "clim"}],
        },
        "extras": {
            "clim_logits": [
                {
                    "name": "unc_clim_raw",
                    "base_col_prefix": "prob_clim_mr_unc",
                    "earlier_col": "prob_clim_mr_unc_earlier",
                },
                {
                    "name": "clim_raw",
                    "base_col_prefix": "prob_clim_mr",
                    "window_start_years": [1900, 2018],
                    "window_end_year": 2024,
                },
            ],
            "forecast_variants": {"base": "", "fixed_cutoff": "_fixed_cutoff"},
            "forecasts": [
                {
                    "name": "fc",
                    "variant": "fixed_cutoff",
                    "raw": True,
                    "calibrated": True,
                    "export_platt_weights": True,
                }
            ],
        },
    }


def dissemination_table(weekly):
    """The units minus one, under the spec's id column, as read from CSV."""
    return pd.DataFrame({"unit": sorted(weekly["id"].unique())[:-1]}).astype(str)


def write_pickle(path, obj):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def assert_same_files(old_dir, new_dir):
    old = {p.name: p.read_bytes() for p in old_dir.iterdir()}
    new = {p.name: p.read_bytes() for p in new_dir.iterdir()}
    assert sorted(new) == sorted(old)
    assert [name for name in old if old[name] != new[name]] == []


def run_both(tmp_path, weekly, spec, dissemination, inputs=None, workers=1):
    """Write the old script's files and evaluate().save()'s, each next to a copy of `inputs`."""
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    for directory in (old_dir, new_dir):
        directory.mkdir()
        for name, obj in (inputs or {}).items():
            write_pickle(directory / name, obj)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _script(spec, weekly, dissemination, str(old_dir), cores=workers)
        evaluation = evaluate(weekly, spec, dissemination, workers=workers, results_dir=new_dir)
    written = evaluation.save(new_dir)
    return evaluation, written, old_dir, new_dir


def test_every_artifact_is_the_scripts(tmp_path):
    years = range(2014, 2026)
    weekly = weekly_table(years)
    spec = cv_spec(
        range(2014, 2024), [2022, 2023], [2025], ["global", "local", "clusters", "bogus"]
    )
    spec["extras"]["external_predictions"] = [{"name": "forest", "file": "forest_preds.pkl"}]
    forest = weekly[weekly["year"].isin([2022, 2023, 2025])][["id", "year", "outcome"]].copy()
    probs = np.random.default_rng(3).dirichlet(np.ones(len(BINS)), size=len(forest))
    for j, b in enumerate(BINS):
        forest[f"cv_{b}"] = probs[:, j]
    # Holdouts other than 2000-2024: the MME reads its saved weights.
    weights = pd.DataFrame([{"objective": "rps", "w_clim_raw": 1.0, "rps": 0.0}])
    inputs = {
        "forest_preds.pkl": forest,
        "mme_optimized_weights_fixed_cutoff_clim_raw_fixed_cutoff_2022_2025.pkl": weights,
    }

    evaluation, written, old_dir, new_dir = run_both(
        tmp_path, weekly, spec, dissemination_table(weekly), inputs
    )

    assert_same_files(old_dir, new_dir)
    assert {path.name for path in written} == {p.name for p in new_dir.iterdir()} - set(inputs)
    assert evaluation.output_tag == "_fixed_cutoff_2022_2025"
    assert set(evaluation.cv_preds) == {
        (model, method)
        for model in ("blend", "clim_only")
        for method in ("global", "local", "clusters")
    }
    assert set(evaluation.coefs_by_year) == {("blend", "global"), ("clim_only", "global")}
    assert set(evaluation.coefs_by_year["blend", "global"]) == {2022, 2023, 2025}
    assert list(evaluation.platt_weights) == [
        "platt_weights_fc_fixed_cutoff_calibrated_df_fixed_cutoff_2022_2025"
    ]
    assert evaluation.mme_weights == {}
    assert {
        "forest",
        "unc_clim_raw",
        "clim_raw",
        "clim_raw_2018_2024",
        "fc_fixed_cutoff_raw",
        "fc_calibrated_fixed_cutoff",
        "mme_fixed_cutoff_clim_raw_opt_rps",
    } <= set(evaluation.cell_metrics["model"])
    # Scoring is on the dissemination units only.
    assert "unit4" not in set(evaluation.cell_metrics["id"])
    assert evaluation.sample_support["rows_excluded"] == 1


def test_mme_weights_are_optimised_when_the_holdouts_are_2000_2024(tmp_path):
    years = range(2000, 2025)
    weekly = weekly_table(years, seed=1, n_cells=3, issues=2)
    spec = cv_spec(years, years, [], methods=[])

    evaluation, _, old_dir, new_dir = run_both(tmp_path, weekly, spec, dissemination_table(weekly))

    assert_same_files(old_dir, new_dir)
    assert list(evaluation.mme_weights) == [
        "mme_optimized_weights_fixed_cutoff_clim_raw_fixed_cutoff_2000_2024"
    ]
    assert evaluation.cv_preds == {} and evaluation.yearly is None


def test_global_cv_in_worker_processes_saves_the_same_files(tmp_path):
    weekly = weekly_table(range(2016, 2024), seed=2)
    spec = cv_spec(range(2016, 2024), [2020, 2021, 2022], [2023], ["global"])
    spec["mme"]["enabled"] = False

    _, _, old_dir, new_dir = run_both(
        tmp_path, weekly, spec, dissemination_table(weekly), workers=2
    )

    assert_same_files(old_dir, new_dir)


def test_mme_of_two_sources_still_fails_at_the_merge():
    # Known defect kept in phase 2: the sources have no lat/lon to merge on.
    weekly = weekly_table(range(2016, 2024))
    spec = cv_spec(range(2016, 2024), [2022], [2023], methods=[])
    spec["mme"]["blend_models"].append(
        {"name": "fc", "source": "forecast", "cal_variant": "fixed_cutoff"}
    )

    with pytest.raises(KeyError, match="lat"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluate(weekly, spec, dissemination_table(weekly))


def test_formula_and_cv_errors_raise_and_baseline_errors_warn():
    weekly = weekly_table(range(2016, 2024))
    spec = cv_spec(range(2016, 2024), [2022], [2023], ["global"])
    spec["mme"]["enabled"] = False
    spec["extras"]["clim_logits"][0]["earlier_col"] = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluation = evaluate(weekly, spec, dissemination_table(weekly))
    messages = [str(w.message) for w in caught]
    assert any(
        m.startswith("Error in clim logit 'unc_clim_raw': unc_clim_raw requires") for m in messages
    )
    assert "unc_clim_raw" not in set(evaluation.cell_metrics["model"])
    # Without the unconditional-climatology baseline there is no pooled summary.
    assert any(m.startswith("Could not compute pooled summary") for m in messages)
    assert evaluation.pooled is None

    spec["models"]["formulas"]["blend"]["text"] = "outcome ~ absent_qx"
    with pytest.raises(ValueError, match="missing predictor columns"):
        evaluate(weekly, spec, dissemination_table(weekly))

    spec["models"]["formulas"]["blend"]["text"] = "outcome ~ prob_clim_mr_qx"
    spec["run"]["training_years"] = []  # no fold has training rows
    with pytest.raises(RuntimeError, match="Error in blend/global"):
        evaluate(weekly, spec, dissemination_table(weekly))


def test_inputs_are_left_unchanged():
    weekly = weekly_table(range(2016, 2024))
    dissemination = dissemination_table(weekly)
    before = weekly.copy(), dissemination.copy()
    spec = cv_spec(range(2016, 2024), [2022], [2023], ["global"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluate(weekly, spec, dissemination)

    pd.testing.assert_frame_equal(weekly, before[0])
    pd.testing.assert_frame_equal(dissemination, before[1])


def test_relative_external_predictions_need_a_results_dir():
    weekly = weekly_table(range(2016, 2024))
    spec = cv_spec(range(2016, 2024), [2022], [2023], ["global"])
    spec["extras"]["external_predictions"] = [{"name": "forest", "file": "preds.pkl"}]

    with pytest.raises(ValueError, match="no results_dir"):
        evaluate(weekly, spec, dissemination_table(weekly))


def test_dissemination_id_column_must_exist():
    weekly = weekly_table(range(2016, 2024))
    spec = cv_spec(range(2016, 2024), [2022], [2023], ["global"])

    with pytest.raises(ValueError, match="cell.id_col 'unit' not found"):
        evaluate(weekly, spec, pd.DataFrame({"adm3_name": ["unit0"]}))
