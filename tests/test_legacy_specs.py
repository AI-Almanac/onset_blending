"""Converting today's stage specs into one BlendConfig."""
import copy
from pathlib import Path

import pytest
import yaml

from onset_blending import BlendConfig, ConfigError
from onset_blending.legacy.specs import convert_spec_files, convert_specs

ROOT = Path(__file__).parents[1]
SPECS = ROOT / "archive" / "specs"

# Each ported config and the spec set it came from.
PORTED = {
    "country_template": dict(
        ground_truth="raw_data/ground_truth_template.yml",
        forecasts={"forecast_model": "raw_data/forecast_template.yml"},
        combine="combine/combine_country_template.yml",
        connect="2025_blend/connect_country_template.yml",
        cv_models="2025_blend/cv_models_country_template.yml",
        regrid="regrid/country_template.yml",
    ),
    "fixed_cutoff_2026": dict(
        ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
        forecasts={"aifs_ens": "raw_data/aifs_ens_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
        combine="combine/combine_template_fixed_cutoff_2026.yml",
        connect="2025_blend/connect_fixed_cutoff_2026.yml",
        cv_models="2025_blend/cv_models_fixed_cutoff_2026.yml",
    ),
    "fixed_cutoff_2026_gencast": dict(
        ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
        forecasts={"gencast": "raw_data/gencast_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
        combine="combine/combine_template_fixed_cutoff_2026_gencast.yml",
        connect="2025_blend/connect_fixed_cutoff_2026_gencast.yml",
        cv_models="2025_blend/cv_models_fixed_cutoff_2026_gencast.yml",
    ),
    "fixed_cutoff_2026_ngcm": dict(
        ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
        forecasts={"ngcm": "raw_data/ngcm_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
        combine="combine/combine_template_fixed_cutoff_2026_ngcm.yml",
        connect="2025_blend/connect_fixed_cutoff_2026_ngcm.yml",
        cv_models="2025_blend/cv_models_fixed_cutoff_2026_ngcm.yml",
    ),
    "fixed_cutoff_one_2026": dict(
        ground_truth="raw_data/ref_rain_fixed_cutoff_2026.yml",
        forecasts={"aifs_ens": "raw_data/aifs_ens_2026.yml", "aifs": "raw_data/aifs_2026.yml"},
        combine="combine/combine_template_fixed_cutoff_2026.yml",
        connect="2025_blend/connect_fixed_cutoff_2026.yml",
        cv_models="2025_blend/cv_models_fixed_cutoff_one_2026.yml",
    ),
}


def spec_paths(name: str) -> dict:
    paths = dict(PORTED[name])
    paths["forecasts"] = {m: SPECS / p for m, p in paths["forecasts"].items()}
    return {k: (SPECS / v if isinstance(v, str) else v) for k, v in paths.items()}


def country_specs() -> dict:
    """The country template set as dicts, to vary."""
    paths = spec_paths("country_template")
    load = lambda p: yaml.safe_load(Path(p).read_text())
    return dict(
        ground_truth=load(paths["ground_truth"]),
        forecasts={m: load(p) for m, p in paths["forecasts"].items()},
        combine=load(paths["combine"]),
        connect=load(paths["connect"]),
        cv_models=load(paths["cv_models"]),
        regrid=load(paths["regrid"]),
    )


def convert(specs: dict):
    return convert_specs(name="test", **specs)


def problems(specs: dict) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        convert(specs)
    return caught.value.problems


@pytest.mark.parametrize("name", sorted(PORTED))
def test_each_ported_config_is_its_spec_set(name):
    converted = convert_spec_files(name=name, data_root=str((ROOT / "archive").resolve()), **spec_paths(name)).config
    assert BlendConfig.from_yaml(ROOT / "configs" / f"{name}.yml") == converted


def test_the_country_template_keeps_every_fact():
    conversion = convert(country_specs())
    config = conversion.config

    assert config.onset.definition.trigger_rule == "first_day_wet"
    assert config.onset.definition.dry_spell.mode == "window_sum"
    assert config.onset.threshold.id_col == "unit_id"
    assert config.observations.files.pattern == r"^rain_(19|20)\d{2}\.nc$"  # the regrid's sources
    assert config.domain.regrid.unit_id_col == "unit_id"
    assert config.domain.centroids is None  # the regrid writes them
    [model] = config.models
    assert (model.lead_days.max, model.rain_horizon.max_day, model.rain_horizon.policy) == (40, 40, "truncate")
    assert [(p.agg, p.window) for p in model.rain_predictors] == [
        ("diff", "trigger"), ("min", "dry_spell"), ("max", "trigger")
    ]
    assert config.features.rain_transform == "fourth_root"
    assert config.climatology.train_years == (2000, 2024)
    assert config.forecast_evaluations()[0].platt_weights


def test_keys_no_code_reads_are_listed_not_kept():
    notes = convert(country_specs()).notes

    for key in ("run.MR", "mme.mc_cores", "options.plus_label_template", "connect: mode"):
        assert any(key in note for note in notes), key


def test_old_defaults_that_differ_are_written_out():
    specs = country_specs()
    del specs["combine"]["options"]["trim_forecasts_after_true_onset"]
    del specs["cv_models"]["extras"]["forecasts"]

    config = convert(specs).config

    assert config.features.trim_after_true_onset is False  # the old combine's default
    assert config.forecast_evaluations() == ()  # nothing scored, as before


def test_an_absent_cutoff_mode_was_ref_onset():
    specs = country_specs()
    del specs["cv_models"]["run"]["cutoff_mode"]

    assert any("'ref_onset' needs onset.reference_onset" in p for p in problems(specs))


def test_restated_facts_must_agree_and_are_reported_together():
    specs = country_specs()
    forecast = specs["forecasts"]["forecast_model"]
    forecast["options"]["window"] = 3
    forecast["thresholds"]["thresh_col"] = "other"
    specs["connect"]["forecast_models"][0]["rain_day_max"] = 35
    specs["cv_models"]["n_bins"] = 5

    found = problems(specs)

    for what in ("the onset definition", "the onset threshold", "rain_day_max", "the number of week bins"):
        assert any(p.startswith(what) or what in p for p in found), what


def test_window_tokens_keep_what_connect_resolved():
    specs = country_specs()
    for spec in (specs["ground_truth"], specs["forecasts"]["forecast_model"]):
        spec["options"]["window"] = 3
    del specs["connect"]["onset_spec"]  # connect then resolved tokens with the defaults (window 5)
    specs["cv_models"]["models"]["formulas"]["blended_model"]["text"] = (
        "outcome ~ prob_clim_qx * diff_forecast_model_qx + max_forecast_model_5day_qx"
    )

    [model] = convert(specs).config.models

    assert [(p.agg, p.window) for p in model.rain_predictors] == [("diff", 5), ("min", "dry_spell"), ("max", 5)]


def test_an_unquoted_window_tag_is_explained():
    specs = country_specs()
    specs["connect"]["climatology"]["window_tags"] = yaml.safe_load("[2000_2024]")

    [found] = [p for p in problems(specs) if "window_tags" in p]
    assert "reads as the number 20002024" in found


def test_window_variants_come_with_their_climatology():
    specs = country_specs()
    specs["cv_models"]["models"]["window_variants"].update(enabled=True, only_if_cutoff_mode=None)
    clim = specs["ground_truth"]["climatologies"]
    clim["clim_2000_2024"] = copy.deepcopy(clim["clim"])
    specs["combine"]["input"]["climatologies"]["clim_2000_2024"] = {"rds": "x.pkl"}

    assert any("needs that climatology" in p for p in problems(specs))

    specs["connect"]["climatology"]["window_tags"] = ["2000_2024"]
    conversion = convert(specs)
    [variant] = conversion.config.evaluation.window_variants
    assert (variant.base, variant.windows, variant.replace) == ("blended_model", ((2000, 2024),), "prob_clim")
    assert conversion.config.climatology.windows == ((2000, 2024),)
    assert any("old replace never matched" in note for note in conversion.notes)


def test_a_custom_climatology_prefix_becomes_prob_clim():
    specs = country_specs()
    specs["connect"]["climatology"]["output_prefix"] = "pc"
    formula = specs["cv_models"]["models"]["formulas"]["blended_model"]
    formula["text"] = formula["text"].replace("prob_clim_qx", "pc_qx")
    for task in specs["cv_models"]["extras"]["clim_logits"]:
        task["base_col_prefix"] = task["base_col_prefix"].replace("prob_clim", "pc")
        if "earlier_col" in task:
            task["earlier_col"] = task["earlier_col"].replace("prob_clim", "pc")

    text = convert(specs).config.evaluation.formulas["blended_model"].text
    assert text.startswith("outcome ~ prob_clim_qx * diff_forecast_model_qx")


def test_dissemination_only_in_training_keeps_that_scope():
    specs = country_specs()
    for spec in (specs["ground_truth"], specs["forecasts"]["forecast_model"]):
        del spec["filter"]["dissemination_cells_file"], spec["filter"]["id_col"]

    domain = convert(specs).config.domain

    assert domain.dissemination.file.endswith("dissemination_units.csv")
    assert domain.dissemination_scope == "training"


def test_combine_names_must_be_what_connect_reads():
    specs = country_specs()
    specs["combine"]["forecasts"]["forecast_model"]["daily"][0]["out"] = "p_onset"

    assert any("connect reads it as 'p_onset_fixed_cutoff'" in p for p in problems(specs))
