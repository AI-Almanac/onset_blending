"""Small BlendConfigs for tests that hand the evaluation a weekly table."""

from onset_blending import BlendConfig


def blend_config(evaluation: dict, *, models=("fc",), bins: int = 2, windows=(), dissemination_id_col=None,
                 grid_ids=None) -> BlendConfig:
    """A config whose models have a diff rain predictor (`diff_<model>_qx`);
    its files are never read."""
    domain = {}
    if dissemination_id_col is not None:
        domain["dissemination"] = {"file": "cells.csv", "id_col": dissemination_id_col}
    if grid_ids is not None:
        domain["grid_ids"] = grid_ids
    return BlendConfig.from_dict({
        "name": "test",
        "domain": domain,
        "observations": {"files": {"folder": "obs"}, "variable": "precip"},
        "models": [{"name": m, "files": {"folder": m}, "rain_predictors": [{"agg": "diff"}]} for m in models],
        "onset": {"threshold": {"mm": 20}},
        "climatology": {"windows": [list(w) for w in windows]},
        "features": {"bins": {"count": bins}},
        "evaluation": evaluation,
    })
