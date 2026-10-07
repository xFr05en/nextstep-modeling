"""Schema and consistency of config/actionability.yaml."""
from src.features import feature_sets
from tests.helpers import load_yaml

ACT = load_yaml("actionability.yaml")
FEATURES = ACT["features"]
REQUIRED = {"label_ko", "label_en", "source", "actionability", "direction", "step", "bounds",
            "difficulty", "months_per_step", "monotone", "protected", "dice_vary", "note_ko", "note_en"}
FLAGS = ["pastdue_special_code", "income_missing", "income_zero", "util_outlier"]


def test_every_model_feature_has_valid_entry():
    for f in feature_sets()["all"]:
        assert f in FEATURES, f"{f} missing from actionability.yaml"
    for name, e in FEATURES.items():
        assert REQUIRED <= set(e), f"{name} missing {REQUIRED - set(e)}"
        assert e.get("model_feature", True) in (True, False), name
        assert e["actionability"] in ACT["classes"], name
        assert e["source"] in {"gmsc", "derived", "simulated"}, name
        assert e["direction"] in {"increase", "decrease", "none"}, name
        assert e["monotone"] in (-1, 0, 1), name
        assert e["label_ko"] and e["label_en"] and e["note_ko"] and e["note_en"], name
        if e["actionability"] in {"IMMUTABLE", "NON_DECREASING", "NOT_RECOMMENDED"}:
            assert e["dice_vary"] is False, f"{name} must be fixed in DiCE"


def test_flags_and_gender_are_immutable():
    for f in FLAGS + ["gender_female"]:
        assert FEATURES[f]["actionability"] == "IMMUTABLE", f
        assert FEATURES[f]["dice_vary"] is False, f
        assert FEATURES[f]["monotone"] == 0, f
    assert FEATURES["gender_female"]["protected"] is True
    assert FEATURES["age"]["protected"] is True and FEATURES["age"]["monotone"] == 0
    for c in load_yaml("data.yaml")["pastdue_columns"]:
        assert FEATURES[c]["actionability"] == "NON_DECREASING", c  # charter: cannot decrease


def test_yaml_consistent_with_other_configs():
    data_cfg, sim_cfg = load_yaml("data.yaml"), load_yaml("simulator.yaml")
    util = FEATURES[data_cfg["util_column"]]
    assert util["bounds"][1] == data_cfg["util_outlier_threshold"]
    for name, spec in sim_cfg["variables"].items():
        assert FEATURES[name]["source"] == "simulated", name
        assert FEATURES[name]["monotone"] == -1 and spec["target_corr"] < 0, name


def test_autopay_is_not_recommended_and_not_a_model_feature():
    """Gameable: removed from paths and from the model, but kept in the simulated data."""
    a = FEATURES["autopay_ratio"]
    assert a["actionability"] == "NOT_RECOMMENDED"
    assert a["dice_vary"] is False and a["model_feature"] is False
    assert a["monotone"] == -1
    assert "autopay_ratio" not in feature_sets()["all"]
    assert "autopay_ratio" in load_yaml("simulator.yaml")["variables"]


def test_all_mission_variables_are_model_features():
    feats = feature_sets()["all"]
    for n in ("telecom_payment_rate", "utility_payment_rate", "spending_consistency",
              "regular_payment_count", "app_login_frequency"):
        assert n in feats, n


def test_mission_variable_classes():
    assert FEATURES["spending_consistency"]["actionability"] == "ACTIONABLE"
    assert FEATURES["spending_consistency"]["dice_vary"] is True
    for n in ("regular_payment_count", "app_login_frequency"):
        assert FEATURES[n]["actionability"] == "NOT_RECOMMENDED" and FEATURES[n]["dice_vary"] is False, n
    for n in ("spending_consistency", "regular_payment_count", "app_login_frequency"):
        assert FEATURES[n]["monotone"] == -1 and FEATURES[n]["source"] == "simulated", n
