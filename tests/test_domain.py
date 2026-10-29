"""Area, dissemination and regrid filters are the legacy nc_utils / sparse transforms'."""
import json

import numpy as np
import pandas as pd
import pytest

from onset_blending import BlendConfig
from onset_blending._legacy import nc_utils
from onset_blending._legacy.geometry_utils import normalize_regrid_weights
from onset_blending._legacy.sparse_transform_utils import compile_sparse_cell_transform, sparse_observed_weighted_mean
from onset_blending.core import domain


def config(tmp_path, **domain_section):
    return BlendConfig.from_dict({
        "name": "t",
        "data_root": str(tmp_path),
        "domain": {"grid_ids": {"decimal_digits": 2, "format": "fixed"}, **domain_section},
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": "m", "files": {"folder": "m"}}],
        "onset": {"threshold": {"mm": 20}},
        "evaluation": {"years": {"training": [2001, 2002], "cv_holdout": [2002]},
                       "formulas": {"b": "outcome ~ prob_clim_qx"}},
    })


def grid():
    lats, lons = np.arange(10.0, 12.01, 0.25), np.arange(70.0, 72.01, 0.25)
    return pd.DataFrame([(a, b) for a in lats for b in lons], columns=["lat", "lon"])


GEOMETRY = {"geometry": {"grid_id_decimal_digits": 2, "grid_id_format": "fixed"}}


def test_a_box_keeps_the_legacy_filters_rows(tmp_path):
    box = {"lat_min": 10.4, "lat_max": 11.6, "lon_max": 71.3}
    area = domain.AreaFilter(config(tmp_path, area_of_interest={"bbox": box}))

    old = nc_utils.filter_by_dissemination_cells(grid(), {"filter": {"bbox": box}, **GEOMETRY})
    pd.testing.assert_frame_equal(area(grid()), old)


def test_dissemination_units_that_restrict_the_inputs_keep_the_legacy_rows(tmp_path):
    cells = grid().sample(20, random_state=1)
    pd.DataFrame({"cell": [f"{a:.2f}_{b:.2f}" for a, b in zip(cells["lat"], cells["lon"])]}).to_csv(
        tmp_path / "cells.csv", index=False)
    area = domain.AreaFilter(config(tmp_path, dissemination={"file": "cells.csv", "id_col": "cell"},
                                    dissemination_scope="inputs"))

    old = nc_utils.filter_by_dissemination_cells(
        grid(), {"filter": {"dissemination_cells_file": str(tmp_path / "cells.csv"), "id_col": "cell"}, **GEOMETRY})
    pd.testing.assert_frame_equal(area(grid()), old)
    assert len(old) == 20


def test_outlines_locate_named_units_by_their_centroids(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "lat": [10.5, 11.5, 13.0], "lon": [70.5, 71.5, 70.0]}).to_csv(
        tmp_path / "centroids.csv", index=False)
    square = {"type": "Polygon", "coordinates": [[[70, 10], [72, 10], [72, 12], [70, 12], [70, 10]]]}
    (tmp_path / "area.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": square}]}))
    area = domain.AreaFilter(config(tmp_path, area_of_interest={"geometry": "area.geojson"},
                                    centroids={"file": "centroids.csv"}))

    kept = area(pd.DataFrame({"id": ["a", "b", "c"]}))
    assert kept["id"].tolist() == ["a", "b"]
    with pytest.raises(ValueError, match="No cells fall inside"):
        area(pd.DataFrame({"id": ["c"]}))


def test_regridding_is_the_legacy_sparse_mean():
    rng = np.random.default_rng(0)
    cells = grid()
    weights = cells.sample(40, random_state=0).assign(
        adm3_name=lambda f: rng.choice(["x", "y", "z"], len(f)), weight=lambda f: rng.uniform(0.1, 1, len(f)))
    regridder = domain.Regridder(weights.rename(columns={"lat": "latitude", "lon": "longitude"}))
    lats, lons = domain.grid_axes(cells)
    transform = regridder.transform(lats, lons)
    values = rng.normal(size=(len(cells), 6))
    values[rng.random(values.shape) < 0.2] = np.nan

    mapping = normalize_regrid_weights(weights.rename(columns={"lat": "latitude", "lon": "longitude"}))
    mapping = mapping.rename(columns={"latitude": "lat", "longitude": "lon"})
    mapping["source_id"] = list(zip(mapping["lat"].astype(float), mapping["lon"].astype(float)))
    sources = tuple((float(a), float(b)) for a in lats for b in lons)
    legacy = compile_sparse_cell_transform(mapping, source_ids=sources, target_ids=tuple(regridder.targets))
    np.testing.assert_array_equal(domain.Regridder.apply(transform, values),
                                  sparse_observed_weighted_mean(legacy, values))
