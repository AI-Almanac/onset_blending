"""Cell transforms on arrays are the legacy table routes: the rows read,
transformed onto the units and searched (nc_utils), rebuilt here as the oracle."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from onset_blending import BlendConfig
from _legacy import nc_utils
import _specs
from onset_blending.core import domain, io, onsets
from onset_blending.core.reference import Reference

from test_pipeline import LEAD_DAYS, write_forecasts, write_observations

WEIGHTS = pd.DataFrame({
    "source_id": ["10.00_70.00", "10.00_70.25", "10.25_70.00", "10.25_70.25", "10.25_70.25"],
    "target_id": ["north", "north", "south", "south", "north"],
    "weight": [0.5, 0.25, 1.0, 0.5, 0.25],
})


def make_config(root, policy="keep", **domain_section):
    WEIGHTS.to_csv(root / "weights.csv", index=False)
    return BlendConfig.from_dict({
        "name": "t",
        "data_root": str(root),
        "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}, **domain_section},
        "observations": {"files": {"folder": "obs"}, "variable": "precip", "missing_rain": policy,
                         "cell_transform": "weights.csv"},
        "models": [{"name": "ens", "files": {"folder": "ens"}, "lead_days": {"max": LEAD_DAYS},
                    "cell_transform": "weights.csv"}],
        "onset": {"threshold": {"mm": 20}, "reference_onset": {"month_day": "06-01"},
                  "definition": {"window": 3, "dry_spell": {"min_dry_days": 5}}},
        "climatology": {"forecast_window": LEAD_DAYS},
        "evaluation": {"years": {"training": "2010:2012", "cv_holdout": "2010:2012"},
                       "formulas": {"b": "outcome ~ prob_clim_qx"}},
    })


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("transform")
    write_observations(root / "obs")
    # Missing rain at one cell for a few days of one season.
    path = root / "obs" / "2005.nc"
    with xr.open_dataset(path) as ds:
        data = ds.load()
    data["precip"][40:45, 0, 1] = np.nan
    data.to_netcdf(path)
    write_forecasts(root / "ens", years=[2011, 2012])
    return root


def legacy_observed(path, context, config, weights_file=None):
    """onsets._observed_file_from_table as it was."""
    spec = _specs.observation_spec(config)
    if weights_file is not None:
        spec = {**spec, "options": {**spec["options"], "cell_transform_file": str(weights_file)}}
    variable = spec["input"]["value_col"]
    df = nc_utils.nc_read_groundtruth_long(
        nc_path=str(path), var_name=variable, dim_rename_map=spec["dimensions"]["rename"],
        missing_rain_policy=spec["options"]["missing_rain_policy"],
    )
    df = domain.with_ids(df, context["convention"])
    df = nc_utils.transform_groundtruth_long(df, nc_utils.read_cell_transform(spec), variable.lower(), spec=spec)
    df = context["area"](df)
    reference = context["reference"]
    return nc_utils.process_ground_truth_rainfall_id(
        df, spec, ref_onset_dt=reference.reference_onset, thr_dt=reference.thresholds, value_col=variable.lower(),
    )["wide"]


@pytest.mark.parametrize("policy", ["keep", "drop", "zero"])
def test_observed_onsets_under_a_cell_transform_are_the_legacy_routes(region, policy):
    config = make_config(region, policy)
    reference = Reference.resolve(config)
    context = onsets.observation_context(config, reference)

    for path in io.list_files(config.observations.files, config, "observations"):
        new = onsets._observed_file(path, context)
        weights = None
        if policy == "drop" and path.name == "2005.nc":
            # The legacy route refused a season whose source series was dropped
            # (the source id went missing); the season's other sources stand in.
            with pytest.raises(ValueError, match="expected weight source IDs are absent"):
                legacy_observed(path, context, config)
            weights = region / "weights_without.csv"
            WEIGHTS[WEIGHTS["source_id"] != "10.00_70.25"].to_csv(weights, index=False)
        old = legacy_observed(path, context, config, weights)
        pd.testing.assert_frame_equal(new.reset_index(drop=True), old.reset_index(drop=True), check_dtype=False)


def legacy_forecast(path, context, config):
    """onsets._forecast_file_from_table as it was, without the member filters."""
    spec = _specs.forecast_spec(config, config.models[0])
    df = nc_utils.nc_read_forecast_wide(
        nc_path=str(path), var_name=context["variable"], dim_rename_map=context["rename"],
        spec=spec, day_dim="day", prefix="rain",
    )
    rain_cols = [c for c in df.columns if c.startswith("rain_day_")]
    if context["unit_scale"] != 1.0:
        df[rain_cols] = df[rain_cols].to_numpy() * context["unit_scale"]
    df = domain.with_ids(df, context["convention"])
    if context["cell_transform"] is not None:
        df = nc_utils.transform_forecast_wide(df, nc_utils.read_cell_transform(spec), spec=spec)
    df = context["area"](df)
    issues = np.sort(pd.to_datetime(df["time"].unique()).to_numpy())
    ids = np.sort(df["id"].astype(str).unique())
    members = np.sort(df["number"].unique())
    days = [int(c.rsplit("_", 1)[1]) for c in rain_cols]
    rain = np.full((len(issues), len(members), len(ids), len(days)), np.nan, dtype=df[rain_cols[0]].dtype)
    t = np.searchsorted(issues, pd.to_datetime(df["time"]).to_numpy())
    m = np.searchsorted(members, df["number"].to_numpy())
    u = np.searchsorted(ids, df["id"].astype(str).to_numpy())
    rain[t, m, u] = df[rain_cols].to_numpy()
    summary = onsets._Summary(ids, issues, days, context)
    summary.add(0, len(issues), rain)
    return summary.dataset()


def forecast_context(config):
    return onsets.forecast_context(config, config.models[0], Reference.resolve(config))


@pytest.mark.parametrize("dissemination", [False, True])
def test_forecast_onsets_under_a_cell_transform_are_the_legacy_routes(region, dissemination):
    pd.DataFrame({"id": ["south"]}).to_csv(region / "cells.csv", index=False)
    extra = {"dissemination": {"file": "cells.csv"}, "dissemination_scope": "inputs"} if dissemination else {}
    config = make_config(region, **extra)
    context = forecast_context(config)

    for path in io.list_files(config.models[0].files, config, "model ens"):
        new, _ = onsets._forecast_file(path, context)
        old = legacy_forecast(path, onsets._file_context(path, context), config)
        xr.testing.assert_identical(new, old)
        assert list(new["unit"].values) == (["south"] if dissemination else ["north", "south"])


def test_dissemination_units_restrict_a_forecasts_points(region):
    cells = ["10.00_70.25", "10.25_70.00"]
    pd.DataFrame({"id": cells}).to_csv(region / "points.csv", index=False)
    config = make_config(region, dissemination={"file": "points.csv"}, dissemination_scope="inputs")
    context = {**forecast_context(config), "cell_transform": None}

    for path in io.list_files(config.models[0].files, config, "model ens"):
        new, _ = onsets._forecast_file(path, context)
        old = legacy_forecast(path, onsets._file_context(path, context), config)
        xr.testing.assert_identical(new.sortby("unit"), old)
        assert sorted(new["unit"].values) == cells
