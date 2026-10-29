"""Today's stage specs, generated from a BlendConfig.

Phase 2 runs the existing stage functions, which read spec dicts. These are
built here from the one config, and only here, so no stage can be configured
differently from another. Phase 3 replaces the functions and this module.
"""

from __future__ import annotations

from .. import config as cfg
from ..config import BlendConfig, ForecastModel

# Dimension names the readers know, mapped to theirs; the config's renames win.
_OBSERVATION_DIMS = {
    "TIME": "time", "time": "time",
    "LATITUDE": "lat", "latitude": "lat", "lat": "lat",
    "LONGITUDE": "lon", "longitude": "lon", "lon": "lon",
    "ADM3_NAME": "adm3_name", "adm3_name": "adm3_name", "adm3": "adm3_name",
}
_FORECAST_DIMS = {
    "time": "time", "day": "day", "lat": "lat", "lon": "lon",
    "adm3_name": "adm3_name", "adm3": "adm3_name", "number": "number", "sample": "number",
}
# The forecast probability series the connect stage bins, by config variant.
_CONNECT_VARIANT = {"fixed_cutoff": "fixed_cutoff", "ref_onset": "ref"}
CLIMATOLOGY = "clim"
UNCONDITIONAL_CLIMATOLOGY = "clim_unc"


def climatology_key(window: tuple[int, int] | None = None) -> str:
    """The combine's name for a climatology (its columns are <key>_p_onset_day_<k>)."""
    return CLIMATOLOGY if window is None else f"{CLIMATOLOGY}_{cfg.window_tag(window)}"


def _onset_options(config: BlendConfig) -> dict:
    definition = config.onset.definition
    spell = definition.dry_spell
    dry_spell = {
        "mode": spell.mode,
        "min_dry_days": spell.min_dry_days,
        "sum_window": spell.sum_window,
        "sum_min_mm": spell.sum_min_mm,
    }
    if spell.dry_day_min_mm is not None:
        dry_spell["dry_day_min_mm"] = spell.dry_day_min_mm
    return {
        "window": definition.window,
        "onset_definition": {
            "wet_day_min_mm": definition.wet_day_min_mm,
            "trigger_rule": definition.trigger_rule,
            "wet_day_comparison": definition.wet_day_comparison,
            "follow_days": definition.follow_days,
            "followup_anchor": definition.followup_anchor,
            "dry_spell": dry_spell,
        },
    }


def _thresholds(config: BlendConfig) -> dict:
    threshold = config.onset.threshold
    if threshold.mm is not None:
        return {"constant": float(threshold.mm)}
    out = {
        "file": str(config.path(threshold.file)),
        "thresh_col": threshold.value_col,
        "lat_col": threshold.lat_col,
        "lon_col": threshold.lon_col,
        "lat_var": threshold.lat_var,
        "lon_var": threshold.lon_var,
        "thresh_var": threshold.value_var,
        "mat_lat_name": threshold.mat_lat_name,
        "mat_lon_name": threshold.mat_lon_name,
        "mat_thresh_name": threshold.mat_value_name,
    }
    if threshold.id_col:
        out["id_col"] = threshold.id_col
    if threshold.scale is not None:
        scale = threshold.scale
        out["rule"] = {"type": "scale", "factor": scale.factor, "offset": scale.offset,
                       "min": scale.min, "max": scale.max}
    return out


def _reference_onset(config: BlendConfig) -> dict | None:
    reference = config.onset.reference_onset
    if reference is None:
        return None
    if reference.month_day is not None:
        return {"constant_month_day": reference.month_day}
    out = {"file": str(config.path(reference.file))}
    for key in ("date_col", "year_col", "day_col", "base_date", "unit_col"):
        if getattr(reference, key) is not None:
            out[key] = getattr(reference, key)
    return out


def _geometry(config: BlendConfig) -> dict:
    ids = config.domain.grid_ids
    if ids.decimal_digits is None:
        return {}
    return {"geometry": {"grid_id_decimal_digits": ids.decimal_digits, "grid_id_format": ids.format}}


def _common(config: BlendConfig) -> dict:
    out = {"thresholds": _thresholds(config), **_geometry(config)}
    reference = _reference_onset(config)
    if reference is not None:
        out["ref_onset"] = reference
    return out


def _cell_transform(config: BlendConfig, path: str | None) -> dict:
    if path is None:
        return {"cell_transform_enabled": False}
    return {"cell_transform_enabled": True, "cell_transform_file": str(config.path(path))}


def observation_spec(config: BlendConfig) -> dict:
    obs = config.observations
    return {
        "type": "ground_truth_rainfall",
        "input": {"value_col": obs.variable},
        "dimensions": {"rename": {**_OBSERVATION_DIMS, **obs.dims}},
        "options": {
            **_onset_options(config),
            "cutoff_month_day": config.onset.season_start,
            "missing_rain_policy": obs.missing_rain,
            # Observation dates moved onto the forecasts' end-of-window labels.
            "date_shift_days": 1 if obs.date_marks == "start" else 0,
            **_cell_transform(config, obs.cell_transform),
        },
        "output": {"write_long": False},
        **_common(config),
    }


def forecast_spec(config: BlendConfig, model: ForecastModel) -> dict:
    options = {
        **_onset_options(config),
        "min_day": model.lead_days.min,
        "max_day": model.lead_days.max,
        "fixed_cutoff_month_day": config.onset.fixed_cutoff,
        **_cell_transform(config, model.cell_transform),
    }
    if model.rain_horizon.max_day is not None:
        options["rain_day_max"] = model.rain_horizon.max_day
        options["rain_horizon_policy"] = model.rain_horizon.resolved_policy
    spec = {
        "type": "rainfall_forecast",
        "input": {"value_col": model.variable, "wide_day_dim": "day", "wide_prefix": "rain"},
        "dimensions": {"rename": {**_FORECAST_DIMS, **model.dims}},
        "options": options,
        **_common(config),
    }
    if model.max_member is not None:
        spec["filter"] = {"max_number": model.max_member}
    return spec


def used_variants(config: BlendConfig) -> list[str]:
    return cfg.probability_variants(config)


def _cutoff_variant(config: BlendConfig) -> str:
    return cfg.cutoff_variant(config)


def connect_spec(config: BlendConfig) -> dict:
    features = config.features
    onset = config.onset.definition
    variants = [_CONNECT_VARIANT[v] for v in used_variants(config)]
    models = []
    for model in config.models:
        entry = {
            "name": model.name,
            "variants": variants,
            "rain_predictors": [
                {"agg": p.agg, "window": cfg.resolve_window(p, onset)} for p in model.rain_predictors
            ],
        }
        if model.rain_transform is not None:
            entry["rain_transform"] = model.rain_transform
        if model.rain_horizon.max_day is not None:
            entry["rain_day_max"] = model.rain_horizon.max_day
            entry["rain_horizon_policy"] = model.rain_horizon.resolved_policy
        models.append(entry)
    return {
        "day_max": features.probability_days,
        "days_per_bin": features.bins.days,
        "n_bins": features.bins.count,
        "climatology": {
            "base_prefix": CLIMATOLOGY,
            "unconditional_prefix": UNCONDITIONAL_CLIMATOLOGY,
            "output_prefix": cfg.CLIMATOLOGY_FEATURE,
            "window_tags": [cfg.window_tag(w) for w in config.climatology.windows],
        },
        "rain_transform": features.rain_transform,
        "forecast_models": models,
        **_geometry(config),
    }


def cv_spec(config: BlendConfig) -> dict:
    evaluation = config.evaluation
    years = evaluation.years
    feature = cfg.CLIMATOLOGY_FEATURE
    baselines = evaluation.climatology_baselines

    clim_logits = [{
        "name": "unc_clim_raw",
        "base_col_prefix": f"{feature}_unc",
        "earlier_col": f"{feature}_unc_earlier",
        "earlier_is_logit": True,
    }]
    # Today's evaluation reads clim_raw as start year 1900, and windowed
    # climatologies as start years sharing one end year per entry.
    starts_by_end: dict[int, list[int]] = {}
    if baselines.conditional:
        starts_by_end[max([*years.training, *years.cv_holdout])] = [1900]
    for start, end in baselines.windows:
        starts_by_end.setdefault(end, []).append(start)
    clim_logits += [
        {"name": "clim_raw", "base_col_prefix": feature, "window_start_years": starts, "window_end_year": end}
        for end, starts in starts_by_end.items()
    ]

    window_variants = []
    for variant in evaluation.window_variants:
        by_end: dict[int, list[int]] = {}
        for start, end in variant.windows:
            by_end.setdefault(end, []).append(start)
        for end, starts in by_end.items():
            entry = {
                "enabled": True,
                "base_name": variant.base,
                "start_years": starts,
                "end_year": end,
                "replace": {"from": f"{variant.replace}_qx", "to": f"{variant.replace}{{window}}_qx"},
            }
            if variant.only_if_cutoff_mode:
                entry["only_if_cutoff_mode"] = variant.only_if_cutoff_mode
            window_variants.append(entry)

    mme = evaluation.mme
    if mme.enabled and mme.optimize == "always":
        raise NotImplementedError("the legacy cv spec cannot express evaluation.mme.optimize 'always'")
    spec = {
        "n_bins": config.features.bins.count,
        "run": {
            "cutoff_mode": evaluation.cutoff_mode,
            "training_years": list(years.training),
            "cv_holdout_years": list(years.cv_holdout),
            "true_holdout_years": list(years.true_holdout),
        },
        **(
            {"cell": {"dissemination": str(config.path(config.domain.dissemination.file)),
                      **({"id_col": config.domain.dissemination.id_col} if config.domain.dissemination.id_col else {})}}
            if config.domain.dissemination is not None
            else {}
        ),
        "cv": {
            "methods": ["cluster" if m == "clusters" else m for m in evaluation.cv_methods],
            "cluster_var": evaluation.cluster_var,
        },
        "models": {
            "formulas": {
                name: {"enabled": formula.enabled, "text": formula.text}
                for name, formula in evaluation.formulas.items()
            },
            "window_variants": window_variants or {"enabled": False},
        },
        "mme": {
            "enabled": mme.enabled,
            "variants": list(mme.variants),
            "blend_models": [
                {"name": s.name, "source": s.source, "cal_variant": s.cal_variant} for s in mme.sources
            ],
            "optimize_if_holdout_is_full_2000_2024": mme.optimize != "never",
            **({"weights_year_tag": mme.weights_year_tag} if mme.weights_year_tag else {}),
        },
        "extras": {
            "clim_logits": clim_logits,
            "forecasts": [
                {
                    "name": s.model,
                    "variant": s.variant or _cutoff_variant(config),
                    "raw": s.raw,
                    "calibrated": s.calibrated,
                    "export_platt_weights": s.platt_weights,
                }
                for s in config.forecast_evaluations()
            ],
            "forecast_variants": {v: cfg.VARIANT_SUFFIX[v] for v in ("base", "fixed_cutoff", "ref_onset")},
            **(
                {"external_predictions": [
                    {"name": e.name, "file": e.file, "method": e.method}
                    for e in evaluation.external_predictions
                ]}
                if evaluation.external_predictions
                else {}
            ),
        },
        **_geometry(config),
    }
    return spec
