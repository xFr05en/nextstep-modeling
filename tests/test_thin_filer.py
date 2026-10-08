"""Thin-filer rules: the mission rule (explicit function) and our proxy; simulator mask mode."""
import numpy as np
import pandas as pd
import pytest

from src.features import add_thin_filer_flag, is_thin_filer_mission, is_thin_filer_proxy
from src.simulator import load_config, simulate
from tests.helpers import ROOT, clean_data, require

CFG = load_config()
COLS = CFG["credit_link"]["pastdue_columns"]
RULE = CFG["thin_filer"]["rule"]


def test_mission_rule_on_handmade_rows():
    df = pd.DataFrame({COLS[0]: [0, np.nan, np.nan, np.nan, 0, 0],
                       COLS[1]: [0, 0, np.nan, np.nan, 0, 0],
                       COLS[2]: [0, 0, 0, np.nan, 0, 0],
                       "card_months": [24, 24, 24, 24, 6, 12]})
    # without card history: only the "2+ missing past-due columns" part applies
    assert list(is_thin_filer_mission(df, COLS)) == [0, 0, 1, 1, 0, 0]
    # with card history: < 12 months also flags (12 itself does not)
    assert list(is_thin_filer_mission(df, COLS, card_history_column="card_months")) == [0, 0, 1, 1, 1, 0]


def test_mission_rule_on_real_file():
    raw = pd.read_csv(require(ROOT / "data" / "raw" / "cs-training.csv"))
    assert is_thin_filer_mission(raw, COLS).sum() == 0                      # raw: no missing past-due values
    clean = clean_data()
    flagged = is_thin_filer_mission(clean, COLS) == 1
    assert flagged.sum() == 269 and (clean.loc[flagged, "pastdue_special_code"] == 1).all()
    assert clean.loc[flagged, "SeriousDlqin2yrs"].mean() == pytest.approx(0.5465, abs=1e-4)


def test_proxy_rule_share_unchanged():
    clean = clean_data()
    assert is_thin_filer_proxy(clean, RULE).mean() == pytest.approx(0.0771, abs=1e-4)
    assert add_thin_filer_flag(clean, RULE)["thin_filer"].equals(is_thin_filer_proxy(clean, RULE))


def test_mask_mode_reaches_requested_share_with_mission_rule():
    sample = clean_data().sample(10_000, random_state=1).reset_index(drop=True)
    sim, info = simulate(sample, CFG, thin_filer_ratio=0.3, thin_filer_mode="mask")
    share = is_thin_filer_mission(sim, COLS).mean()
    assert abs(share - 0.30) <= 0.005
    assert len(sim) == len(sample)                                          # masking keeps every row
    assert info["scenario"]["thin_rule"] == "mission" and sim["thin_filer"].mean() == pytest.approx(share)


def test_mode_validation_and_default():
    sample = clean_data().head(500)
    with pytest.raises(ValueError, match="thin_filer_mode"):
        simulate(sample, CFG, thin_filer_ratio=0.3, thin_filer_mode="drop")
    with pytest.raises(ValueError, match="needs a thin_filer_ratio"):
        simulate(sample, CFG, thin_filer_mode="mask")
    assert CFG["scenario"]["thin_filer_mode"] == "subsample"
