"""Scenario checks for the simulator parameters thin_filer_ratio and bias_ratio.

Writes reports/simulator_scenarios.json. Each bias level is a fresh simulation with the final
setup retrained in 5-fold CV; approvals use out-of-fold PD at the score cutoff (475).

Run from the project root:  python notebooks/simulator_scenarios.py   (about 6 minutes)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import add_thin_filer_flag, feature_sets  # noqa: E402
from src.scoring import load_config as load_scoring, pd_to_score  # noqa: E402
from src.data.simulator import load_config, simulate  # noqa: E402
from src.train import cross_validate, load_yaml, make_folds, monotone_vector  # noqa: E402

TARGET = "SeriousDlqin2yrs"
MISSION = [n for n, v in load_config()["variables"].items() if v.get("shared_factor")]


def group_stats(approved: np.ndarray, y: np.ndarray, groups: pd.Series) -> dict:
    rows = {}
    for g in groups.dropna().unique():
        m = (groups == g).to_numpy()
        rows[str(g)] = {"n": int(m.sum()), "default_rate": round(float(y[m].mean()), 4),
                        "approval_rate": round(float(approved[m].mean()), 4),
                        "approval_if_repaid": round(float(approved[m & (y == 0)].mean()), 4),
                        "approval_if_defaulted": round(float(approved[m & (y == 1)].mean()), 4)}
    rates = [r["approval_rate"] for r in rows.values()]
    rep = [r["approval_if_repaid"] for r in rows.values()]
    dft = [r["approval_if_defaulted"] for r in rows.values()]
    return {"groups": rows,
            "approval_gap": round(max(rates) - min(rates), 4),
            "di_ratio_min_over_max": round(min(rates) / max(rates), 4),
            "eo_gap": round(max(max(rep) - min(rep), max(dft) - min(dft)), 4)}


def main() -> None:
    cfg = load_config()
    tcfg = load_yaml("train.yaml")
    scoring = load_scoring()
    cutoff = min(g["min_score"] for g in scoring["grades"] if g["grade"] in scoring["approval"]["approve_grades"])
    clean = add_thin_filer_flag(pd.read_csv(ROOT / load_yaml("data.yaml")["processed_path"]), cfg["thin_filer"]["rule"])
    feats = feature_sets()["all"]
    bands = cfg["age_bands"]
    out = {"thin_filer_ratio": {}, "bias_ratio": {}}

    for ratio in (None, 0.15, 0.30):
        sim, info = simulate(clean, cfg, thin_filer_ratio=ratio)
        out["thin_filer_ratio"][str(ratio)] = {**info["scenario"],
                                               "mission_observed_r": {n: info["corr_target"][n] for n in MISSION}}
        print("thin", ratio, info["scenario"])

    for bias in (0.0, 0.2, 0.5):
        sim, info = simulate(clean, cfg, bias_ratio=bias)
        folds = make_folds(sim, tcfg)
        _, oof = cross_validate(sim, folds, tcfg, "xgb", feats, "none", monotone_vector(feats))
        approved = (pd_to_score(oof, scoring) >= cutoff).astype(int)
        y = sim[TARGET].to_numpy()
        gender = sim["gender_female"].map({0: "male", 1: "female"})
        band = pd.cut(sim["age"], bands["edges"], labels=bands["labels"]).astype(str)
        out["bias_ratio"][str(bias)] = {
            "scenario": info["scenario"],
            "mission_latent_link_a": {n: info["params"][n]["a"] for n in MISSION},
            "mission_observed_r": {n: info["corr_target"][n] for n in MISSION},
            "gender": group_stats(approved, y, gender),
            "age_band": group_stats(approved, y, band),
        }
        r = out["bias_ratio"][str(bias)]
        print(f"bias {bias}: gender gap {r['gender']['approval_gap']} DI {r['gender']['di_ratio_min_over_max']} "
              f"EO {r['gender']['eo_gap']} | age gap {r['age_band']['approval_gap']} "
              f"DI {r['age_band']['di_ratio_min_over_max']} EO {r['age_band']['eo_gap']}")

    (ROOT / "reports" / "simulator_scenarios.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
