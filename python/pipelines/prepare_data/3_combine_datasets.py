#!/usr/bin/env python3
# ==============================================================================
# Script: 3_combine_datasets.py
# ==============================================================================
# Purpose
#   Combine climatology + forecast families + ground truth into one combined
#   dataset for modeling / downstream processing.
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/prepare_data/3_combine_datasets.py --spec_id combine_template_2025
# ==============================================================================

import argparse
import os
import sys
import pickle

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from onset_blending._legacy.misc import coalesce
from onset_blending._legacy.read_spec import load_spec
from onset_blending._legacy.combine_forecasts_utils import (
    read_ground_truth_wide,
    read_and_format_climatology_wide,
    format_forecast_family,
)


def main():
    parser = argparse.ArgumentParser(description="Combine datasets into wide table.")
    parser.add_argument("-i", "--spec_id", required=True,
                        help="Spec ID (loads specs/combine/<id>.yml)")
    args = parser.parse_args()

    spec_id = args.spec_id
    spec = load_spec(spec_id, "combine")

    # Read ground truth
    truth = read_ground_truth_wide(spec["input"]["ground_truth_wide_rds"])

    # Read climatologies
    clim_list = []
    if spec["input"].get("climatologies"):
        for nm, x in spec["input"]["climatologies"].items():
            clim_list.append(
                read_and_format_climatology_wide(
                    path=x["rds"],
                    out_prefix=coalesce(x.get("out_prefix"), f"{nm}_p_onset"),
                )
            )
    elif spec["input"].get("climatology_rds"):
        clim_list.append(
            read_and_format_climatology_wide(
                path=spec["input"]["climatology_rds"],
                out_prefix="clim_p_onset",
            )
        )

    # Read forecast families
    forecast_parts = {
        nm: format_forecast_family(nm, conf)
        for nm, conf in spec.get("forecasts", {}).items()
    }
    forecast_daily = {nm: fp["daily"] for nm, fp in forecast_parts.items()}
    forecast_const = {nm: fp["constants"] for nm, fp in forecast_parts.items()}


#    # === DIAGNOSTIC: check id/time coverage vs forecasts ===
#    if clim_list and forecast_daily:
#        clim_ids  = set(clim_list[0]["id"].astype(str).unique())
#        fcast_ids = set(list(forecast_daily.values())[0]["id"].astype(str).unique())
#        only_in_fcast = fcast_ids - clim_ids
#        only_in_clim  = clim_ids - fcast_ids
#        print(f"\nDIAGNOSTIC: id coverage")
#        print(f"  IDs in forecast but NOT in clim : {len(only_in_fcast)}")
#        print(f"  IDs in clim but NOT in forecast : {len(only_in_clim)}")
#        if only_in_fcast:
#            print(f"  Sample forecast-only ids: {list(only_in_fcast)[:5]}")
#        if only_in_clim:
#            print(f"  Sample clim-only ids   : {list(only_in_clim)[:5]}")
#    
#        # Check time overlap for a shared id
#        shared_id = next(iter(clim_ids & fcast_ids), None)
#        if shared_id:
#            clim_times  = set(clim_list[0].loc[clim_list[0]["id"]==shared_id, "time"].astype(str))
#            fcast_times = set(list(forecast_daily.values())[0].loc[
#                list(forecast_daily.values())[0]["id"]==shared_id, "time"].astype(str))
#            print(f"\n  Time overlap check for id='{shared_id}':")
#            print(f"    clim  time range : {sorted(clim_times)[:3]} ... {sorted(clim_times)[-3:]}")
#            print(f"    fcast time range : {sorted(fcast_times)[:3]} ... {sorted(fcast_times)[-3:]}")
#    # === END DIAGNOSTIC ===


    # Combine daily: clim + forecast daily tables
    daily_tables = clim_list + list(forecast_daily.values())

    join_type = "outer" if spec.get("options", {}).get("join") == "full" else "inner"

    daily_wide = daily_tables[0]
    for tbl in daily_tables[1:]:
        daily_wide = daily_wide.merge(tbl, on=["id", "time", "year"], how=join_type)

    # Add year from time if needed
    daily_wide["year"] = pd.to_datetime(daily_wide["time"]).dt.year

    # Merge ground truth
    daily_wide = daily_wide.merge(truth, on=["id", "year"], how="left")

    # Optional trimming
    if spec.get("options", {}).get("trim_forecasts_after_true_onset", False):
        mask = daily_wide["true_onset_date"].isna() | (
            pd.to_datetime(daily_wide["time"]) <= pd.to_datetime(daily_wide["true_onset_date"])
        )
        daily_wide = daily_wide[mask]

    # Merge constants
    if forecast_const:
        const_tables = list(forecast_const.values())
        const_all = const_tables[0]
        for tbl in const_tables[1:]:
            const_all = const_all.merge(tbl, on=["id", "time", "year"], how="outer")
        daily_wide = daily_wide.merge(const_all, on=["id", "time", "year"], how="left")

    # Write output
    out_dir = spec["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    #out_pkl = os.path.join(out_dir, f"{spec_id}_combined_wide.pkl")
    basename = spec.get("output", {}).get("basename", spec_id)
    out_pkl = os.path.join(out_dir, f"{basename}_combined_wide.pkl")

    with open(out_pkl, "wb") as f:
        pickle.dump(daily_wide, f)

    print(f"Wrote combined wide dataset: {out_pkl}")
    print(f"Rows: {len(daily_wide)}, Cols: {len(daily_wide.columns)}")


if __name__ == "__main__":
    main()
