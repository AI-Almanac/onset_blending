"""
3_fit_final_model.py
====================
Fits the blending model on ALL available training years (no holdout),
and saves the resulting coefficient bundle for use in future-year prediction.

A command-line wrapper over onset_blending.model.fit_final().

This is the "production" model — unlike the CV models in 1_blend_evaluation.py
which each hold out one year, this uses every year of data to fit the best
possible model for forecasting future (unseen) years.

The saved bundle is identical in format to what 1_blend_evaluation.py saves
per holdout year, so apply_blend_model.py can load it directly via --coef_tag.

Usage (run from repo root)
--------------------------
    python 3_fit_final_model.py \\
        --spec_id  cv_models_fixed_cutoff \\
        --model    blended_model \\
        [--input_path Monsoon_Data/Processed_Data/pipeline_input/cv_data_fixed_cutoff_new_pipeline.pkl] \\
        [--method  global] \\
        [--tag     final] \\
        [--out_dir Monsoon_Data/results/2025_model_evaluation/]

Output
------
    coefs_blended_model_global_final.pkl   <- coef bundle (coefs, scaler, features, formula)
    coefs_blended_model_global_final.csv   <- human-readable coefficient table

Then to apply to a future year:
    python apply_blend_model.py \\
        --spec_id    cv_models_fixed_cutoff \\
        --model      blended_model \\
        --year       2026 \\
        --coef_tag   final \\
        --input_path Monsoon_Data/Processed_Data/pipeline_input/wide_2026.pkl
"""

import os
import sys
# --- MODIFICATION FOR SUBFOLDER SUPPORT ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
# ------------------------------------------
import pickle
import argparse
import logging
import pandas as pd

from onset_blending._legacy.blend_evaluation_utils import (
    DEFAULT_PIPELINE_INPUT_ROOT,
    resolve_blend_input_path,
)
from onset_blending._legacy.read_spec import load_spec
from onset_blending.evaluation import RunSettings
from onset_blending.model import fit_final


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_coef_table(coef_df, model_name, label):
    """Pretty-print the coefficient table."""
    print(f"\n{'='*65}")
    print(f"Coefficients: {model_name}  |  {label}")
    print(f"{'='*65}")

    classes  = sorted(coef_df["class"].unique())
    features = coef_df["feature"].unique()

    header = f"  {'feature':<32}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    print("  " + "-" * (32 + 12 * len(classes)))

    for feat in features:
        row = f"  {feat:<32}"
        for cls in classes:
            val = coef_df[
                (coef_df["class"] == cls) & (coef_df["feature"] == feat)
            ]["coefficient"]
            row += f"{float(val.iloc[0]):>12.4f}" if len(val) > 0 else f"{'N/A':>12}"
        print(row)

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
        description="Fit blending model on all years and save coef bundle for future prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spec_id", required=True,
                        help="CV spec ID (e.g. cv_models_fixed_cutoff)")
    parser.add_argument("--model",   required=True,
                        help="Model name in cv_models*.yml (e.g. blended_model)")
    parser.add_argument("--method",  default="global",
                        help="CV method (default: global)")
    parser.add_argument("--tag",     default="final",
                        help="Tag for output filenames (default: final). "
                             "Output: coefs_{model}_{method}_{tag}.pkl")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory. "
                             "Defaults to Monsoon_Data/results/2025_model_evaluation/")
    parser.add_argument("--input_path", default=None,
                        help="Read the connector pickle at this exact path "
                             "(highest priority).")
    parser.add_argument("--dissem_file", default=None,
                        help="Override the dissemination CSV configured by "
                             "cell.dissemination")
    parser.add_argument("--pipeline_input_dir", default=DEFAULT_PIPELINE_INPUT_ROOT,
                        help="Root directory containing the spec-configured "
                             "pipeline input subdirectory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    # ── Load spec ──────────────────────────────────────────────────────
    spec = load_spec(args.spec_id, "2025_blend")
    run_cfg = spec.get("run", {})
    run = RunSettings.from_spec(spec)
    fit_years = run.fit_years
    out_dir = args.out_dir or "Monsoon_Data/results/2025_model_evaluation"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load input data ────────────────────────────────────────────────
    input_path = resolve_blend_input_path(
        run.cutoff_mode,
        run_cfg,
        input_path=args.input_path,
        pipeline_input_root=args.pipeline_input_dir,
    )
    print(f"\nLoading wide_df from: {input_path}")
    dissemination_csv = (
        args.dissem_file
        or spec["cell"].get("dissemination")
        or "Monsoon_Data/dissemination_cells.csv"
    )
    dissemination = pd.read_csv(dissemination_csv, dtype=str)

    # ── Fit model on all training years ───────────────────────────────
    # The table goes straight in, so only fit_final's copies of it stay alive.
    model = fit_final(
        _read_pickle(input_path), spec, dissemination, model_name=args.model
    )
    bundle = model.to_legacy_bundle()

    # ── Print and save ────────────────────────────────────────────────
    print_coef_table(
        bundle["coefs"],
        args.model,
        f"all years ({min(fit_years)}-{max(fit_years)})",
    )

    stem     = f"coefs_{args.model}_{args.method}_{args.tag}"
    pkl_path = os.path.join(out_dir, f"{stem}.pkl")
    csv_path = os.path.join(out_dir, f"{stem}.csv")

    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    bundle["coefs"].to_csv(csv_path, index=False)

    print(f"Saved:")
    print(f"  Coef bundle (pkl) -> {pkl_path}")
    print(f"  Coef table  (csv) -> {csv_path}")
    print(f"\nTo apply to a future year:")
    print(f"  python apply_blend_model.py \\")
    print(f"      --spec_id    {args.spec_id} \\")
    print(f"      --model      {args.model} \\")
    print(f"      --year       <future_year> \\")
    print(f"      --coef_tag   {args.tag} \\")
    print(f"      --input_path <path_to_future_wide_df.pkl>")


if __name__ == "__main__":
    main()
