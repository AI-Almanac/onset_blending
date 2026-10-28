"""BlendConfig: one source of truth, validated as a whole."""
import copy
from pathlib import Path

import pytest
import yaml

from onset_blending import BlendConfig, ConfigError
from onset_blending import config as cfgmod

EXAMPLE = Path(__file__).parents[1] / "configs" / "india_aifs.yml"


def example() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def problems(data: dict) -> list[str]:
    with pytest.raises(ConfigError) as caught:
        BlendConfig.from_dict(data)
    return caught.value.problems


def test_the_example_loads_and_round_trips():
    config = BlendConfig.from_yaml(EXAMPLE)

    assert config.model_names == ("aifs_ensv2", "aifsv2")
    assert config.evaluation.years.holdout[:5] == (2021, 2022, 2023, 2024, 2025)
    assert cfgmod.output_tag(config) == "_fixed_cutoff_2000_2025"
    assert BlendConfig.from_dict(config.to_dict()) == config
    assert BlendConfig.from_dict(yaml.safe_load(config.to_yaml())) == config


def test_names_and_windows_are_derived_from_one_place():
    config = BlendConfig.from_yaml(EXAMPLE)
    onset = config.onset.definition

    assert cfgmod.resolve_window(cfgmod.RainPredictor(agg="max", window="trigger"), onset) == 3
    assert cfgmod.resolve_window(cfgmod.RainPredictor(agg="min", window="min_dry"), onset) == 5
    assert cfgmod.rain_feature_base("m", cfgmod.RainPredictor(agg="min", window=10), onset) == "min_m_10day"
    assert cfgmod.probability_feature_base("m", "fixed_cutoff") == "m_p_onset_fixed_cutoff"
    assert cfgmod.outcome_labels(config) == ["week1", "week2", "week3", "week4", "later"]
    assert cfgmod.climatology_baseline_names(config) == ["unc_clim_raw", "clim_raw"]


@pytest.mark.parametrize(
    ("text", "years"),
    [("2000:2003", (2000, 2001, 2002, 2003)), ("2000:2001,2005", (2000, 2001, 2005)),
     ([2003, 2001, 2001], (2001, 2003)), (2010, (2010,)), ("2000..2002", (2000, 2001, 2002))],
)
def test_years_are_read_in_every_usual_form(text, years):
    data = example()
    data["evaluation"]["years"]["cv_holdout"] = text
    assert BlendConfig.from_dict(data).evaluation.years.cv_holdout == years


def test_a_typo_in_a_key_is_an_error_with_a_suggestion():
    data = example()
    data["onset"]["thresold"] = data["onset"].pop("threshold")

    found = problems(data)

    assert any("onset.thresold: unknown key (did you mean 'threshold'?)" in p for p in found)
    assert any("onset.threshold: required" in p for p in found)


def test_a_formula_can_only_use_features_the_config_builds():
    data = example()
    data["evaluation"]["formulas"]["blended_model"] = "outcome ~ prob_clim_qx * diff_aifs_ens_v2_qx"

    [found] = [p for p in problems(data) if "formulas" in p]

    assert "'diff_aifs_ens_v2_qx' is not a feature" in found
    assert "diff_aifs_ensv2_qx" in found  # the suggestion


def test_formula_functions_are_not_mistaken_for_features():
    data = example()
    data["evaluation"]["formulas"]["blended_model"] = (
        "outcome ~ I(prob_clim_week1 ** 2) + np.log(onset_threshold) + diff_aifsv2_qx"
    )
    BlendConfig.from_dict(data)


def test_every_problem_is_reported_at_once():
    data = example()
    data["features"] = {"bins": {"count": 5, "days": 7}, "probability_days": 28}
    data["evaluation"]["cutoff_mode"] = "ref_onset"
    del data["onset"]["reference_onset"]
    data["models"][1]["rain_predictors"].append({"agg": "diff", "window": 5})

    found = problems(data)

    assert any("5 bins of 7 days need 35 probability days" in p for p in found)
    assert any("'ref_onset' needs onset.reference_onset" in p for p in found)
    assert any("one 'diff' predictor per model" in p for p in found)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["models"][0].update(lead_days={"min": 1, "max": 20}), "below features.probability_days"),
        (lambda d: d["models"][0].update(rain_horizon={"max_day": 40}, lead_days={"min": 2, "max": 45}),
         "needs lead_days.min of 1"),
        (lambda d: d["models"][0].update(rain_horizon={"policy": "exact"}), "needs a max_day"),
        (lambda d: d["models"][0].update(lead_days={"min": 1, "max": 30}), "read 38 lead days"),
        (lambda d: d["onset"].update(threshold={"mm": 20, "file": "t.csv"}), "exactly one of 'mm' or 'file'"),
        (lambda d: d["observations"]["files"].update(paths=["a.nc"]), "either 'folder'"),
        (lambda d: d["climatology"].update(horizons=[{"start": "05-01", "end": "06-01", "forecast_window": 30}]),
         "'forecast_window' or 'horizons'"),
        (lambda d: d["climatology"].update(forecast_window=20), "below features.probability_days"),
        (lambda d: d["onset"].update(season_start="5/1"), "expected a month-day"),
        (lambda d: d["evaluation"].update(climatology_baselines={"unconditional": False}), "cannot be turned off"),
        (lambda d: d["models"].append(copy.deepcopy(d["models"][0])), "names must be unique"),
        (lambda d: d["evaluation"].update(cv_methods=["neighbors"]), "is not one of"),
        (lambda d: d.setdefault("features", {}).update(rain_transform="cube"), "is not identity"),
    ],
)
def test_cross_section_rules(change, message):
    data = example()
    change(data)
    assert any(message in p for p in problems(data)), problems(data)


def test_windowed_climatologies_are_declared_once_and_referenced():
    data = example()
    data["climatology"]["windows"] = ["1990:2020", [2000, 2020]]
    data["evaluation"]["climatology_baselines"] = {"windows": ["2000:2020"]}
    data["evaluation"]["window_variants"] = [{"base": "blended_model", "windows": ["1990:2020"]}]
    data["evaluation"]["formulas"]["windowed"] = "outcome ~ prob_clim_1990_2020_qx"

    config = BlendConfig.from_dict(data)

    assert config.climatology.windows == ((1990, 2020), (2000, 2020))
    assert "clim_raw_2000_2020" in cfgmod.climatology_baseline_names(config)
    data["evaluation"]["window_variants"][0]["windows"] = ["1980:2020"]
    assert any("not in climatology.windows" in p for p in problems(data))


def test_mme_sources_must_be_scored():
    data = example()
    data["evaluation"]["mme"] = {"enabled": True, "sources": [
        {"name": "clim_raw", "source": "clim"},
        {"name": "aifsv2", "source": "forecast", "cal_variant": "fixed_cutoff"},
    ]}
    BlendConfig.from_dict(data)

    data["evaluation"]["mme"]["sources"][1]["cal_variant"] = "ref_onset"
    data["evaluation"]["mme"]["sources"].append({"name": "clim_raw_1990_2020", "source": "clim"})
    found = problems(data)
    assert any("needs evaluation.forecasts to calibrate 'aifsv2' (ref_onset)" in p for p in found)
    assert any("'clim_raw_1990_2020' is not a scored climatology baseline" in p for p in found)
