#!/usr/bin/env python3
# ==============================================================================
# Script: 1_blend_evaluation.py
# ==============================================================================
# Purpose
#   Cross-validate weekly-bin multinomial onset models using a YAML spec.
#   Saves metrics, reliability data, Platt weights, and MME blend weights.
#   A command-line wrapper over onset_blending.evaluation.evaluate().
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/blending_process/1_blend_evaluation.py --spec_id cv_models
#   python pipelines/blending_process/1_blend_evaluation.py --spec_id hindcast_1965_1978
# ==============================================================================

import argparse
import logging
import os
import sys
import pickle
import yaml
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from onset_blending._legacy.misc import coalesce
from onset_blending._legacy.blend_evaluation_utils import (
    print_formula_summary,
    resolve_blend_input_path,
)
from onset_blending.evaluation import evaluate


def _read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Cross-validate weekly-bin blending models.")
    parser.add_argument("--spec_id", default="cv_models",
                        help="Spec file name (without .yml) in specs/2025_blend/")
    parser.add_argument("--cores", type=int, default=None,
                        help="Process workers for global CV folds")
    parser.add_argument("--work_dir", default=None,
                        help="If provided, input pkl is read directly from this directory, "
                             "overriding pipeline_input_dir in the spec.")
    parser.add_argument("--input_path", default=None,
                        help="Read the connector pickle at this exact path (highest priority).")
    parser.add_argument("--results_dir", default=None,
                        help="If provided, all outputs are written to this directory, "
                             "overriding pipeline_output_dir in the spec.")
    args = parser.parse_args()
    if args.cores is not None and args.cores < 1:
        parser.error("--cores must be a positive integer")
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    spec_path = os.path.join("specs", "2025_blend", f"{args.spec_id}.yml")
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    if args.results_dir:
        path_box = args.results_dir
    else:
        pipeline_output_dir = spec["run"].get("pipeline_output_dir", "")
        path_box = os.path.join("Monsoon_Data/results", pipeline_output_dir)
    os.makedirs(path_box, exist_ok=True)

    cutoff_mode = coalesce(spec.get("run", {}).get("cutoff_mode"), "ref_onset")
    input_path = resolve_blend_input_path(
        cutoff_mode,
        spec.get("run"),
        input_path=args.input_path,
        work_dir=args.work_dir,
    )
    print(f"Reading {input_path}")
    dissemination = pd.read_csv(spec["cell"].get("dissemination", ""), dtype=str)

    # The table goes straight in: evaluate() keeps only the rows it uses, and
    # a reference held here would keep the whole table alive for the run.
    evaluation = evaluate(
        _read_pickle(input_path),
        spec,
        dissemination,
        workers=args.cores or 1,
        results_dir=path_box,
    )
    print_formula_summary(spec, cutoff_mode, formulas=evaluation.formulas)
    evaluation.save(path_box)
    print("\nDone.")


if __name__ == "__main__":
    main()
