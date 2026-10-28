import os
import sys
# --- MODIFICATION FOR SUBFOLDER SUPPORT ---
# Identify the repository root (one level up from this script's directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
# ------------------------------------------
"""
apply_blend_model.py
====================
Postprocessing script: reads saved blending model coefficients for a given
year, applies them to a wide_df pickle, and writes per-year output files.

Two modes:
  1. Historical holdout year: loads from the standard pipeline input and uses
     the configured true/CV holdout years to resolve the coefficient filename.
  2. Operational year: loads from --input_path and skips metrics because no
     ground truth is available.

Usage (run from repo root)
--------------------------
    # Historical year:
    python predict/apply_blend_model.py \\
        --spec_id   cv_models_fixed_cutoff \\
        --model     blended_model \\
        --year_tag  2022 \\
        --year      2022 \\
        [--coef_dir Monsoon_Data/results/2025_model_evaluation/] \\
        [--out_dir  Monsoon_Data/results/2025_model_evaluation/per_year/] \\
        [--pipeline_input_dir Monsoon_Data/Processed_Data/pipeline_input] \\
        [--dissem_file Monsoon_Data/dissemination_cells.csv]

    # Future year (no ground truth):
    python predict/apply_blend_model.py \\
        --spec_id    cv_models_fixed_cutoff \\
        --model      blended_model \\
        --year       2026 \\
        --coef_tag   clim_mok_date_2022_year2022 \\
        --input_path Monsoon_Data/Processed_Data/pipeline_input/wide_2026.pkl

Input files
------------
1. The spec YAML (via --spec_id):
   specs/2025_blend/cv_models_fixed_cutoff.yml
   Used to get cutoff_mode, holdout_years, and the formula.

2. The coef pickle (looked up automatically from --coef_dir):
   coefs_blended_model_global_clim_mok_date_2022_year2022.pkl  (historical)
   coefs_blended_model_global_final.pkl                         (future, via --coef_tag)
   Contains fitted coefficients, scaler, feature names, and (for new bundles)
   the resolved training formula.

3. The wide pickle (historical) or --input_path (future):
   Monsoon_Data/Processed_Data/pipeline_input/cv_data_clim_mok_date_new_pipeline.pkl
   The full feature dataset — patsy builds the design matrix from this.

4. Dissemination cells (historical metrics only):
   Configured by cell.dissemination in the blend spec; --dissem_file can
   override that path.

Output files
------------
    blended_model_global_year2022_preds.csv   <- predicted cv_* probs per row
    blended_model_global_year2022_preds.pkl
    blended_model_global_year2022_coefs.csv   <- coefficient table (printed + saved)
    blended_model_global_year2022_metrics.csv <- Brier/RPS/AUC (historical only)
"""

import pickle
import argparse
import warnings
import numpy as np
import pandas as pd


from onset_blending._legacy.blend_evaluation_utils import (
    DEFAULT_PIPELINE_INPUT_ROOT,
    build_formulas_from_spec,
    compute_cell_metrics_fast,
    configured_holdout_years,
    make_cutoff_tag,
    make_year_tag,
    resolve_blend_input_path,
)
from onset_blending._legacy.read_spec import load_spec
from onset_blending._legacy.nc_utils import ensure_id_col
from onset_blending._legacy.spatial_id_utils import (
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from onset_blending.model import FittedModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_coefs(coef_path):
    """Load coefficient bundle (coefs, scaler, features, optional formula).
    Falls back gracefully if old-format bare DataFrame pickle is found.
    """
    if not os.path.exists(coef_path):
        raise FileNotFoundError(f"Coefficient file not found: {coef_path}")
    with open(coef_path, "rb") as f:
        obj = pickle.load(f)
    # backward compat: old pickles are bare DataFrames
    if isinstance(obj, pd.DataFrame):
        warnings.warn(
            "Coef file is old format (no scaler/features). "
            "Re-run 1_blend_evaluation.py or 3_fit_final_model.py to get full bundle."
        )
        return {
            "coefs":    obj,
            "scaler":   None,
            "features": list(obj["feature"].unique()),
        }
    return obj


def load_dissemination_cells(spec, path_override, authoritative_ids):
    """Load dissemination cells under the shared spatial-ID contract."""
    cell_cfg = spec.get("cell") or {}
    path = (
        path_override
        or cell_cfg.get("dissemination")
        or "Monsoon_Data/dissemination_cells.csv"
    )
    cells = pd.read_csv(path, dtype=str)
    configured_id_col = cell_cfg.get("id_col") or cell_cfg.get(
        "dissemination_id_col"
    )
    if configured_id_col:
        if configured_id_col not in cells.columns:
            raise ValueError(
                f"cell.id_col '{configured_id_col}' not found. "
                f"Found: {cells.columns.tolist()}"
            )
        cells = cells.rename(columns={configured_id_col: "id"})
    convention = None
    if "id" not in cells.columns and "adm3_name" not in cells.columns:
        convention = resolve_grid_id_convention(
            spec=spec,
            authoritative_ids=authoritative_ids,
            context="prediction/dissemination ID handoff",
        )
    cells = ensure_id_col(
        cells,
        spec=spec,
        convention=convention,
        context="prediction dissemination IDs",
    )
    validate_id_coordinate_consistency(
        cells, context="prediction dissemination IDs"
    )
    return cells.drop_duplicates("id")


def resolve_application_formula(coef_bundle, spec, cutoff_mode, model_name,
                                data):
    """Prefer the saved training formula; support older bundles via the spec."""
    saved_formula = coef_bundle.get("formula")
    if saved_formula:
        return saved_formula, "saved coefficient bundle"

    formulas = build_formulas_from_spec(spec, cutoff_mode, data=data)
    if model_name not in formulas:
        raise ValueError(
            f"Model '{model_name}' not found in spec formulas. "
            f"Available: {list(formulas.keys())}"
        )
    return formulas[model_name], "current spec (legacy coefficient bundle)"


def print_coef_table(coef_df, model_name, label):
    """Pretty-print the coefficient table."""
    print(f"\n{'='*65}")
    print(f"Coefficients: {model_name}  |  {label}")
    print(f"{'='*65}")

    classes = sorted(coef_df["class"].unique())
    features = coef_df["feature"].unique()

    # Header
    header = f"  {'feature':<32}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    print("  " + "-" * (32 + 12 * len(classes)))

    # One row per feature
    for feat in features:
        row = f"  {feat:<32}"
        for cls in classes:
            val = coef_df[(coef_df["class"] == cls) & (coef_df["feature"] == feat)]["coefficient"]
            row += f"{float(val.iloc[0]):>12.4f}" if len(val) > 0 else f"{'N/A':>12}"
        print(row)

    # Intercept row
    row = f"  {'[intercept]':<32}"
    for cls in classes:
        val = coef_df[coef_df["class"] == cls]["intercept"].iloc[0]
        row += f"{float(val):>12.4f}"
    print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply saved blending model coefficients to produce per-year forecasts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spec_id", required=True,
                        help="CV spec ID (e.g. cv_models_fixed_cutoff)")
    parser.add_argument("--model",   required=True,
                        help="Model name as in cv_models*.yml (e.g. blended_model)")
    parser.add_argument("--year",    required=True, type=int,
                        help="Test year to apply the model for")
    parser.add_argument("--year_tag", default=None,
                    help="Override year tag in filename, e.g. '2000_2022'. "
                         "If not provided, derived from configured true and CV "
                         "holdout years in spec.")
    parser.add_argument("--method",  default="global",
                        help="CV method (default: global)")
    parser.add_argument("--coef_tag",   default=None,                        # ← NEW
                        help="Override entire output tag for coef filename. "  # ← NEW
                             "e.g. fixed_cutoff_2022_year2022  "
                             "for coefs_blended_model_global_fixed_cutoff_2022_year2022.pkl"
                             "e.g. 'final' for coefs saved by 3_fit_final_model.py. "  # ← NEW
                             "Coef file becomes: coefs_{model}_{method}_{coef_tag}.pkl")  # ← NEW
    parser.add_argument("--input_path", default=None,                        # ← NEW
                        help="Path to wide_df pickle for a future year. "     # ← NEW
                             "e.g. Monsoon_Data/Processed_Data/2026/cv_data... next line "
                             "cv_data_fixed_cutoff_new_pipeline_2026.pkl"
                             "If provided, skips standard pipeline input lookup "  # ← NEW
                             "and skips outcome filtering and metrics.")       # ← NEW
    parser.add_argument("--coef_dir", default=None,
                        help="Directory containing coef_*.pkl files. "
                             "e.g. Monsoon_Data/results/dry_spell_aifs_aifs_ens/"
                             "Defaults to Monsoon_Data/results/2025_model_evaluation/")
    parser.add_argument("--out_dir",  default=None,
                        help="Output directory. "
                             "Defaults to Monsoon_Data/results/2025_model_evaluation/per_year/")
    parser.add_argument("--dissem_file", default=None,
                        help="Override the dissemination CSV configured by "
                             "cell.dissemination")
    parser.add_argument("--pipeline_input_dir", default=DEFAULT_PIPELINE_INPUT_ROOT,
                        help="Root directory containing the spec-configured "
                             "pipeline input subdirectory")
    args = parser.parse_args()

    # ── Load spec ──────────────────────────────────────────────────────
    spec = load_spec(args.spec_id, "2025_blend")
    cutoff_mode     = spec["run"]["cutoff_mode"]
    holdout_years   = configured_holdout_years(spec)
    training_years  = [int(y) for y in spec["run"]["training_years"]]

    is_future_year = args.input_path is not None                              # ← NEW

    cutoff_tag  = make_cutoff_tag(cutoff_mode)
    #year_tag    = make_year_tag(holdout_years)
    if args.year_tag:
        year_tag = f"_{args.year_tag}"          # e.g. "2000_2022" → "_2000_2022"
    else:
        year_tag = make_year_tag(holdout_years) # derived from spec as before
    output_tag  = f"{cutoff_tag}{year_tag}"

    # ── Paths ──────────────────────────────────────────────────────────
    coef_dir = args.coef_dir or "Monsoon_Data/results/2025_model_evaluation"
    out_dir  = args.out_dir  or os.path.join(coef_dir, "per_year")
    os.makedirs(out_dir, exist_ok=True)

    if args.coef_tag:                                                         # ← NEW
        # e.g. --coef_tag final -> coefs_blended_model_global_final.pkl      # ← NEW
        coef_filename = f"coefs_{args.model}_{args.method}_{args.coef_tag}.pkl"  # ← NEW
    else:
        coef_filename = f"coefs_{args.model}_{args.method}{output_tag}_year{args.year}.pkl"
    coef_path     = os.path.join(coef_dir, coef_filename)
    print(f"\nLoading coefs from: {coef_path}")

    # ── Load coefficients ─────────────────────────────────────────────
    # ← NEW: load_coefs now returns a dict bundle instead of a bare DataFrame
    coef_bundle  = load_coefs(coef_path)                                      # ← NEW
    coef_df      = coef_bundle["coefs"]                                       # ← NEW
    feature_cols = coef_bundle["features"]                                    # ← NEW

    label = "final model (all years)" if args.coef_tag == "final" else f"test_year={args.year}"  # ← NEW
    print_coef_table(coef_df, args.model, label)

    # Save coefficient table as CSV
    coef_csv = os.path.join(out_dir, f"{args.model}_{args.method}_year{args.year}_coefs.csv")
    coef_df.to_csv(coef_csv, index=False)
    print(f"Coefficients saved → {coef_csv}")

    # ── Load input data ────────────────────────────────────────────────
    if is_future_year:                                                        # ← NEW
        # Future year: load from --input_path, no outcome filtering           # ← NEW
        print(f"\nLoading future year data from: {args.input_path}")          # ← NEW
        with open(args.input_path, "rb") as f:                               # ← NEW
            test_df = pickle.load(f)                                          # ← NEW
        if not isinstance(test_df, pd.DataFrame):                            # ← NEW
            test_df = pd.DataFrame(test_df)                                  # ← NEW
        if "outcome" not in test_df.columns:                                  # ← NEW
            test_df["outcome"] = np.nan   # no ground truth yet               # ← NEW
        if "year" not in test_df.columns:                                     # ← NEW
            test_df["year"] = args.year                                       # ← NEW
    else:
        # Historical year: load from standard pipeline input
        input_path = resolve_blend_input_path(
            cutoff_mode,
            spec.get("run"),
            pipeline_input_root=args.pipeline_input_dir,
        )
        print(f"\nLoading wide_df from: {input_path}")
        with open(input_path, "rb") as f:
            wide_df = pickle.load(f)
        wide_df = wide_df[wide_df["outcome"].notna()].copy()  # filter only for historical

        # Filter to test year only
        test_df = wide_df[wide_df["year"] == args.year].copy()
        if test_df.empty:
            raise ValueError(f"No rows found for year {args.year} in {input_path}")

    print(f"Input rows for year {args.year}: {len(test_df)}")

    # ── Get formula ────────────────────────────────────────────────────
    formula_str, formula_source = resolve_application_formula(
        coef_bundle, spec, cutoff_mode, args.model, test_df
    )
    model = FittedModel.from_legacy_bundle(coef_bundle, formula=formula_str)

    print(f"\nFormula ({formula_source}): {formula_str}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── Apply the model → predicted probabilities ─────────────────────
    # The design matrix comes from the saved formula (patsy expands a:b
    # interaction terms exactly as in training) and the saved scaler.
    print(f"\nApplying coefficients to {len(test_df)} test rows...")
    cv_preds = test_df.join(model.predict(test_df), how="left")

    cv_cols = [c for c in cv_preds.columns if c.startswith("cv_")]
    display_cols = ["time", "id", "year"] + (["outcome"] if not is_future_year else []) + cv_cols
    print(f"\nFirst 5 predictions:")
    print(cv_preds[display_cols].head().to_string(index=False))

    # ── Output stem (used by both metrics and save blocks) ─────────────
    stem = f"{args.model}_{args.method}_year{args.year}"

    # ── Compute metrics (historical only) ──────────────────────────────
    if not is_future_year and cv_preds["outcome"].notna().any():              # ← NEW
        cv_preds = ensure_id_col(
            cv_preds, spec=spec, context="prediction IDs"
        )
        dissemination_cells = load_dissemination_cells(
            spec, args.dissem_file, cv_preds["id"]
        )
        valid_ids = set(cv_preds["id"].dropna().unique())
        overlap = valid_ids & set(dissemination_cells["id"])
        print(f"Matching dissemination cell ids: {len(overlap)}")

        metrics = compute_cell_metrics_fast(cv_preds, allowed_cells=dissemination_cells)
        metrics["model"]      = args.model
        metrics["cv_method"]  = args.method
        metrics["test_year"]  = args.year

        all_row = metrics[metrics["id"] == "ALL"].iloc[0]
        print(f"\nMetrics for year {args.year} (pooled ALL cells):")
        print(f"  Brier : {all_row['brier']:.4f}")
        print(f"  RPS   : {all_row['rps']:.4f}")
        print(f"  AUC   : {all_row['auc']:.4f}")
        print(f"  n     : {int(all_row['n'])}")

        met_csv = os.path.join(out_dir, f"{stem}_metrics.csv")
        metrics.to_csv(met_csv, index=False)
        print(f"  Metrics           → {met_csv}")
    else:                                                                     # ← NEW
        print("\nNo outcome data — skipping metrics (future year prediction)")  # ← NEW

    # ── Save predictions ───────────────────────────────────────────────
    pred_csv = os.path.join(out_dir, f"{stem}_preds.csv")
    pred_pkl = os.path.join(out_dir, f"{stem}_preds.pkl")
    cv_preds.to_csv(pred_csv, index=False)
    with open(pred_pkl, "wb") as f:
        pickle.dump(cv_preds, f)

    print(f"\nOutputs saved:")
    print(f"  Predictions (CSV) → {pred_csv}")
    print(f"  Predictions (pkl) → {pred_pkl}")
    print(f"  Coefficients      → {coef_csv}")


if __name__ == "__main__":
    main()
