"""Alternative-data simulator: reproducibility, latent correlation rule, bounds."""
import copy
from functools import lru_cache

import numpy as np
import pandas as pd

from src.simulator import load_config, simulate
from tests.helpers import clean_data

CFG = load_config()
NAMES = list(CFG["variables"])


@lru_cache
def _sample() -> pd.DataFrame:
    return clean_data().sample(20_000, random_state=0).reset_index(drop=True)


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


def test_latent_correlation_within_charter_range():
    _, info = _run()
    lo, hi = CFG["target_corr_range"]
    for n in NAMES:
        r = abs(info["corr_target_latent"][n])
        assert lo <= r <= hi, f"{n}: latent |r| = {r}"
        assert abs(r - abs(CFG["variables"][n]["target_corr"])) <= 0.01, n


def test_gender_independent_bounds_and_inter_correlation():
    sim, info = _run()
    assert abs(info["gender_corr_target"]) < 0.03
    inter = pd.DataFrame(info["inter_corr"]).to_numpy()
    assert np.abs(inter[~np.eye(len(NAMES), dtype=bool)]).max() <= CFG["max_inter_corr"]
    assert sim[["telecom_ontime_rate", "utility_ontime_rate", "autopay_ratio"]].stack().between(0, 1).all()
    assert sim["insurance_paid_months"].between(0, 24).all()
    cap = (sim["age"] - CFG["tenure_min_age"]).clip(lower=0) * 12
    assert (sim["telecom_tenure_months"] <= cap).all()
    assert sim[NAMES].notna().all().all()
