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
        assert FEATURES[name]["source"] == "simulated"
        assert FEATURES[name]["monotone"] == -1 and spec["target_corr"] < 0, name


def test_autopay_is_not_recommended_but_still_constrained():
    """Intentional exception: removed from paths, but monotone -1 is kept so the model is unchanged."""
    a = FEATURES["autopay_ratio"]
    assert a["actionability"] == "NOT_RECOMMENDED"
    assert a["dice_vary"] is False
    assert a["monotone"] == -1
    assert "autopay_ratio" in feature_sets()["all"]
