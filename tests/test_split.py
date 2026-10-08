"""Stratified 70:15:15 split and test-set evaluation."""
import json

import pandas as pd
import pytest

from src.data.preprocessor import make_split
from tests.helpers import ROOT, load_yaml, require, sim_data

CFG = load_yaml("train.yaml")
TARGET = CFG["target"]
MISSION = ["telecom_payment_rate", "utility_payment_rate", "spending_consistency",
           "regular_payment_count", "app_login_frequency"]


def _labels():
    return make_split(sim_data(), CFG)


def test_split_ratios_are_70_15_15():
    sp = CFG["split"]
    assert (sp["train"], sp["validation"], sp["test"]) == (0.70, 0.15, 0.15)
    counts = _labels().value_counts()
    n = counts.sum()
    for part in ("train", "validation", "test"):
        assert abs(counts[part] - sp[part] * n) <= 1, (part, counts[part])


def test_split_is_stratified_disjoint_complete_and_reproducible():
    df, labels = sim_data(), _labels()
    rates = df[TARGET].groupby(labels).mean()
    assert (rates - df[TARGET].mean()).abs().max() < 0.001       # within 0.1 percentage point
    assert labels.index.equals(df.index) and labels.notna().all()  # every row in exactly one part
    assert set(labels.unique()) == {"train", "validation", "test"}
    assert labels.equals(make_split(df, CFG))


def test_saved_split_file_matches():
    saved = pd.read_csv(require(ROOT / CFG["outputs"]["split"]))
    assert list(saved.columns) == ["row_id", "split"]
    assert saved.set_index("row_id")["split"].equals(_labels().rename("split").rename_axis("row_id"))


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_mission_variables_observed_corr_on_each_split(part):
    df, labels = sim_data(), _labels()
    sub = df[labels == part]
    corr = sub[MISSION + [TARGET]].corr()[TARGET]
    for n in MISSION:
        assert 0.30 <= abs(corr[n]) <= 0.50, f"{part}/{n}: {corr[n]:.4f}"


def test_final_summary_reports_test_set_once():
    s = json.loads(require(ROOT / CFG["outputs"]["final_summary"]).read_text())
    assert s["trained_on"] == "train+validation" and s["evaluated_on"] == "test"
    t = s["test"]
    for k in ("auc", "ks", "thin_auc_lift", "psi_trainval_vs_test", "precision", "recall", "f1"):
        assert k in t, k
    lo, hi = t["thin_auc_lift_ci95"]
    assert lo <= t["thin_auc_lift"] <= hi
    assert s["monotonic_loss_validation"]["evaluated_on"] == "validation"
    assert all(v for v in s["charter"].values() if v is not None)
