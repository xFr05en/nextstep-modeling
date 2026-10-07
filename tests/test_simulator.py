"""Alternative-data simulator: reproducibility, correlation rules, shared factor, bounds."""
import copy
from functools import lru_cache

import numpy as np
import pandas as pd

from src.simulator import credit_score, load_config, simulate
from tests.helpers import clean_data, sim_data

CFG = load_config()
NAMES = list(CFG["variables"])
MISSION = ["telecom_payment_rate", "utility_payment_rate", "spending_consistency",
           "regular_payment_count", "app_login_frequency"]
TARGET = "SeriousDlqin2yrs"


@lru_cache
def _sample() -> pd.DataFrame:
    return clean_data().sample(10_000, random_state=0).reset_index(drop=True)


@lru_cache
def _run(seed: int = CFG["seed"]):
    cfg = copy.deepcopy(CFG)
    cfg["seed"] = seed
    return simulate(_sample(), cfg)


def test_same_seed_identical_different_seed_differs():
    a, _ = _run()
    b, _ = simulate(_sample(), CFG)  # fresh call, not cached
    pd.testing.assert_frame_equal(a, b)
    c, _ = _run(CFG["seed"] + 1)
    assert not a[NAMES].equals(c[NAMES])


def test_mission_variable_names_exist_and_are_observed_calibrated():
    for n in MISSION:
        assert n in CFG["variables"], n
        assert CFG["variables"][n]["calibrate_on"] == "observed"
        assert CFG["variables"][n].get("shared_factor") is True
    assert "telecom_ontime_rate" not in CFG["variables"] and "utility_ontime_rate" not in CFG["variables"]


def test_mission_variables_observed_corr_on_full_data():
    """The checklist verifies with df.corr(): observed |r| in [0.30, 0.50] on the generated data."""
    sim = sim_data()
    corr = sim[MISSION + [TARGET]].corr()[TARGET]
    lo, hi = CFG["target_corr_range"]
    for n in MISSION:
        assert lo <= abs(corr[n]) <= hi, f"{n}: observed |r| = {abs(corr[n]):.4f}"
        assert abs(corr[n] - CFG["variables"][n]["target_corr"]) <= CFG["calibration"]["tolerance_observed"] + 1e-9


def test_correlation_rules_on_sample():
    _, info = _run()
    lo, hi = CFG["target_corr_range"]
    for n in NAMES:
        spec = CFG["variables"][n]
        r = info["corr_target"][n] if spec["calibrate_on"] == "observed" else info["corr_target_latent"][n]
        assert lo <= abs(r) <= hi, f"{n}: |r| = {abs(r)}"
        tol = CFG["calibration"]["tolerance_observed" if spec["calibrate_on"] == "observed" else "tolerance_latent"]
        assert abs(r - spec["target_corr"]) <= tol + 1e-9, n
    assert info["max_pair_corr"] <= CFG["max_inter_corr"]
    assert info["shared_factor"]["feasible"]


def test_gender_independent_bounds():
    sim, info = _run()
    assert abs(info["gender_corr_target"]) < 0.03
    rates = ["telecom_payment_rate", "utility_payment_rate", "autopay_ratio"]
    assert sim[rates].stack().between(0, 1).all()
    assert sim["insurance_paid_months"].between(0, 24).all()
    assert sim["spending_consistency"].between(0, 100).all()
    for n, hi in [("regular_payment_count", 20), ("app_login_frequency", 30)]:
        assert sim[n].between(0, hi).all() and (sim[n] == sim[n].round()).all(), n
    cap = (sim["age"] - CFG["tenure_min_age"]).clip(lower=0) * 12
    assert (sim["telecom_tenure_months"] <= cap).all()
    assert sim[NAMES].notna().all().all()


def test_credit_part_has_no_age_input():
    """app_login_frequency must not be linked to age: the credit score ignores the age column."""
    df = _sample()
    shuffled = df.copy()
    shuffled["age"] = np.random.default_rng(0).permutation(df["age"].to_numpy())
    np.testing.assert_array_equal(credit_score(df, CFG), credit_score(shuffled, CFG))
    assert not CFG["variables"]["app_login_frequency"].get("age_cap")
