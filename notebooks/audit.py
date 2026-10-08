"""Audit of the shipped model: recourse, fairness, stability, sensitivity (official 70/15/15 split).

Changes nothing: configs, the versioned final model and the training MLflow runs stay as they are.
Outputs: reports/audit/*.json and one run per check in the MLflow experiment `nextstep-audit`.
The official split replaces the earlier 80/20 holdout re-selection.

Run from the project root:  python notebooks/audit.py   (about 12 minutes)
"""
from __future__ import annotations

import copy
import itertools
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.data.preprocessor import make_split  # noqa: E402
from src.data.simulator import load_config as load_sim, simulate  # noqa: E402
from src.features import add_thin_filer_flag, feature_sets  # noqa: E402
from src.scoring import load_config as load_scoring, pd_to_score  # noqa: E402
from src.train import (cross_validate, load_yaml, make_estimator, make_folds, monotone_vector,  # noqa: E402
                       setup_mlflow, use_experiment)

OUT = ROOT / "reports" / "audit"
CFG = load_yaml("train.yaml")
ACT = load_yaml("actionability.yaml")["features"]
SIM_CFG = load_sim()
SCORING = load_scoring()
CUTOFF = min(g["min_score"] for g in SCORING["grades"] if g["grade"] in SCORING["approval"]["approve_grades"])
TARGET_SCORE = SCORING["recourse_target_score"]
FS = feature_sets()
FEATS = FS["all"]
TARGET, THIN = CFG["target"], CFG["thin_col"]
SUMMARY = json.loads((ROOT / CFG["outputs"]["final_summary"]).read_text())
FINAL = SUMMARY["winner"]
FINAL_MONO = monotone_vector(FEATS, tuple(FINAL["dropped_constraints"]))
MISSION = [n for n, v in SIM_CFG["variables"].items() if v.get("shared_factor")]
STABILITY_SEEDS = [11, 22, 33]
EO_LIMIT, DI_LOW, DI_HIGH = 0.10, 0.80, 1.25


def save(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def scores(est, X: pd.DataFrame) -> np.ndarray:
    return pd_to_score(est.predict_proba(X)[:, 1], SCORING)


def log_audit_run(name: str, metrics: dict, payload: dict) -> None:
    use_experiment(CFG, "audit")
    with mlflow.start_run(run_name=f"audit__{name}"):
        mlflow.set_tags({"step": "7", "stage": "audit", "owner": "wonbin", "evaluated_on": "test"})
        mlflow.log_params({"model_file": SUMMARY["model_file"], "model_version": SUMMARY["model_version"]})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.log_dict(payload, f"{name}.json")


# =====================================================================
# 1. Recourse feasibility (rejected test-set applicants, shipped model)
# =====================================================================

def vary_specs() -> dict:
    """Model features DiCE may vary, with direction, step, bounds and months per step from the YAML."""
    specs = {}
    for f in FEATS:
        e = ACT[f]
        if not e["dice_vary"]:
            continue
        step = e["step"]
        mult = isinstance(step, str) and step.endswith("%")
        specs[f] = {"dir": 1 if e["direction"] == "increase" else -1, "mult": mult,
                    "step": float(step.rstrip("%")) / 100 if mult else float(step),
                    "lo": e["bounds"][0], "hi": e["bounds"][1],
                    "mps": float(e["months_per_step"]), "class": e["actionability"]}
    return specs


def apply_steps(x: np.ndarray, spec: dict, n) -> np.ndarray:
    """Value after n steps in the allowed direction, clipped to bounds. NaN stays NaN."""
    n = np.asarray(n, float)
    y = x * (1 + spec["step"]) ** n if spec["mult"] else x + spec["dir"] * spec["step"] * n
    lo = -np.inf if spec["lo"] is None else spec["lo"]
    hi = np.inf if spec["hi"] is None else spec["hi"]
    y = np.clip(y, lo, hi)
    return np.where(spec["dir"] > 0, np.maximum(y, x), np.minimum(y, x))


def max_steps(spec: dict, horizon: int) -> float:
    return np.floor(horizon / spec["mps"] + 1e-9)


def corner_scores(est, X, specs, horizon, subsets) -> dict:
    out = {}
    for sub in subsets:
        Xm = X.copy()
        for v in sub:
            Xm[v] = apply_steps(X[v].to_numpy(), specs[v], max_steps(specs[v], horizon))
        out[sub] = scores(est, Xm)
    return out


def minimal_paths(est, X, chosen, specs, horizon, target, iters: int = 14) -> pd.DataFrame:
    """Smallest uniform fraction t of the max change (per chosen subset) that reaches `target`."""
    lo, hi = np.zeros(len(X)), np.ones(len(X))

    def build(t):
        Xm = X.copy()
        for v, s in specs.items():
            use = np.array([v in c for c in chosen])
            if use.any():
                new = apply_steps(X[v].to_numpy(), s, np.ceil(t * max_steps(s, horizon)))
                Xm.loc[use, v] = new[use]
        return Xm

    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = scores(est, build(mid)) >= target
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    return build(hi)


def recourse_view(est, X_rej, thin_rej, specs, horizon, label, target):
    names = list(specs)
    subsets = [c for k in (1, 2, 3) for c in itertools.combinations(names, k)]
    sc = corner_scores(est, X_rej, specs, horizon, subsets + [tuple(names)])
    all_vars = sc.pop(tuple(names)) >= target
    best_k = np.zeros(len(X_rej), int)
    chosen: list = [None] * len(X_rej)
    best_score = np.full(len(X_rej), -1)
    for k in (1, 2, 3):
        open_rows = best_k == 0
        for sub in [s for s in subsets if len(s) == k]:
            better = (sc[sub] >= target) & open_rows & (sc[sub] > best_score)
            for i in np.flatnonzero(better):
                chosen[i] = sub
            best_score = np.where(better, sc[sub], best_score)
        best_k = np.where(open_rows & (best_score >= target), k, best_k)
    feasible = best_k > 0
    top = pd.Series([" + ".join(c) for c in chosen if c]).value_counts().head(5)
    res = {"view": label, "horizon_months": horizon, "target_score": target, "variables": names,
           "rejected": int(len(X_rej)), "feasible_share": round(float(feasible.mean()), 4),
           "feasible_share_thin": round(float(feasible[thin_rej].mean()), 4) if thin_rej.any() else None,
           "feasible_share_not_thin": round(float(feasible[~thin_rej].mean()), 4),
           "not_feasible": int((~feasible).sum()),
           "vars_needed": {str(k): int((best_k == k).sum()) for k in (1, 2, 3)},
           "blocked_by_3_var_limit": int((~feasible & all_vars).sum()),
           "blocked_even_with_all_vars": int((~feasible & ~all_vars).sum()),
           "top_paths": {k: int(v) for k, v in top.items()}}
    return res, feasible, chosen


def blocker_profile(X_rej, score_rej, blocked, target) -> dict:
    sub = X_rej[blocked]
    pdc = load_yaml("data.yaml")["pastdue_columns"]
    serious = (sub[pdc[1]].fillna(0) > 0) | (sub[pdc[2]].fillna(0) > 0)
    return {"rows": int(blocked.sum()),
            "median_score": float(np.median(score_rej[blocked])),
            "median_points_short": float(np.median(target - score_rej[blocked])),
            "share_serious_late_60plus": round(float(serious.mean()), 4),
            "share_special_code_96_98": round(float(sub["pastdue_special_code"].mean()), 4)}


def build_paths(est, X_rej, thin_rej, feas, chosen, specs, target) -> dict:
    idx = np.flatnonzero(feas)
    Xf = X_rej.iloc[idx].reset_index(drop=True)
    ch = [chosen[i] for i in idx]
    X_corner = Xf.copy()
    for i, sub in enumerate(ch):
        for v in sub:
            X_corner.loc[i, v] = apply_steps(np.array([Xf.loc[i, v]]), specs[v], max_steps(specs[v], 12))[0]
    X_min = minimal_paths(est, Xf, ch, specs, 12, target)
    thin = thin_rej[idx]
    return {"corner": {"X": X_corner, "thin": thin, "deployed": scores(est, X_corner)},
            "minimal": {"X": X_min, "thin": thin, "deployed": scores(est, X_min)}}


def check_recourse(test: pd.DataFrame, est) -> tuple[dict, dict]:
    X = test[FEATS].astype(float).reset_index(drop=True)
    s0 = scores(est, X)
    rej = s0 < CUTOFF
    X_rej, thin_rej, s_rej = X[rej].reset_index(drop=True), test[THIN].to_numpy()[rej] == 1, s0[rej]
    specs = vary_specs()
    no_time = {k: v for k, v in specs.items() if v["class"] != "TIME_ONLY"}
    results, paths = {}, {}
    for label, sp, h, tgt in [("main_12m_495", specs, 12, TARGET_SCORE), ("all_24m_495", specs, 24, TARGET_SCORE),
                              ("no_time_only_12m_495", no_time, 12, TARGET_SCORE),
                              ("reference_12m_475", specs, 12, CUTOFF), ("reference_24m_475", specs, 24, CUTOFF)]:
        res, feas, chosen = recourse_view(est, X_rej, thin_rej, sp, h, label, tgt)
        if (~feas).any():
            res["not_feasible_profile"] = blocker_profile(X_rej, s_rej, ~feas, tgt)
        if label in ("main_12m_495", "reference_12m_475"):
            prefix = "target495" if label == "main_12m_495" else "target475"
            for kind, p in build_paths(est, X_rej, thin_rej, feas, chosen, sp, tgt).items():
                paths[f"{prefix}_{kind}"] = p
            res["minimal_path_reaches_target"] = round(float((paths[f"{prefix}_minimal"]["deployed"] >= tgt).mean()), 4)
        results[label] = res
        print(f"[recourse] {label}: feasible {res['feasible_share']:.4f} (thin {res['feasible_share_thin']})")
    results.update({"test_rows": int(len(X)), "rejected": int(rej.sum()), "rejected_share": round(float(rej.mean()), 4),
                    "rejected_share_thin": round(float(rej[test[THIN].to_numpy() == 1].mean()), 4)})
    return results, paths


# =====================================================================
# 2. Fairness on the shipped model's test-set decisions (fixed score-475 rule)
# =====================================================================

def group_table(approved, y, groups: pd.Series) -> dict:
    rows = {}
    for g in sorted(groups.dropna().unique()):
        m = (groups == g).to_numpy()
        p = approved[m].mean()
        rows[str(g)] = {"n": int(m.sum()), "default_rate": round(float(y[m].mean()), 4),
                        "approval_rate": round(float(p), 4),
                        "approval_ci95": [round(float(p - 1.96 * np.sqrt(p * (1 - p) / m.sum())), 4),
                                          round(float(p + 1.96 * np.sqrt(p * (1 - p) / m.sum())), 4)],
                        "tpr_approval_if_repaid": round(float(approved[m & (y == 0)].mean()), 4),
                        "fpr_approval_if_defaulted": round(float(approved[m & (y == 1)].mean()), 4)}
    rates = [r["approval_rate"] for r in rows.values()]
    tpr = [r["tpr_approval_if_repaid"] for r in rows.values()]
    fpr = [r["fpr_approval_if_defaulted"] for r in rows.values()]
    di = min(rates) / max(rates)
    out = {"groups": rows, "di_ratio_min_over_max": round(di, 4),
           "eo_tpr_diff": round(max(tpr) - min(tpr), 4), "eo_fpr_diff": round(max(fpr) - min(fpr), 4)}
    out["eo_gap"] = max(out["eo_tpr_diff"], out["eo_fpr_diff"])
    out["di_pass"] = bool(DI_LOW <= di <= DI_HIGH)
    out["eo_pass"] = bool(out["eo_gap"] <= EO_LIMIT)
    return out


def check_fairness(test: pd.DataFrame) -> dict:
    sc = pd.read_csv(ROOT / CFG["outputs"]["test_scores"]).set_index("row_id").loc[test.index]
    approved, y = sc["approved"].to_numpy(), sc["default"].to_numpy()
    bands = SIM_CFG["age_bands"]
    groups = {"gender": test["gender_female"].map({0: "male", 1: "female"}),
              "age_band": pd.cut(test["age"], bands["edges"], labels=bands["labels"]).astype(str)}
    out = {k: group_table(approved, y, g) for k, g in groups.items()}
    out["rule"] = {"di_band": [DI_LOW, DI_HIGH], "eo_limit": EO_LIMIT,
                   "positive_for_tpr_fpr": "approved; TPR = approval rate among repaid, FPR = among defaulted"}
    for k in ("gender", "age_band"):
        print(f"[fairness] {k}: DI {out[k]['di_ratio_min_over_max']} (pass {out[k]['di_pass']}), "
              f"EO TPR {out[k]['eo_tpr_diff']} FPR {out[k]['eo_fpr_diff']} (pass {out[k]['eo_pass']})")
    return out


# =====================================================================
# 3. Stability: retrain on 80% subsamples of train+validation, score the test set
# =====================================================================

def train_variant(tv: pd.DataFrame, seed: int, params: dict | None = None):
    sub = tv.sample(frac=0.8, random_state=seed)
    cfg = copy.deepcopy(CFG)
    if params:
        cfg["models"][FINAL["model"]].update(params)
    y = sub[TARGET].to_numpy()
    return make_estimator(FINAL["model"], FEATS, FINAL["resampling"], cfg, y, FINAL_MONO,
                          FINAL["calibrated"]).fit(sub[FEATS].astype(float), y)


def check_stability(tv, test, deployed, paths) -> dict:
    X_te, y_te = test[FEATS].astype(float), test[TARGET].to_numpy()
    ref = scores(deployed, X_te)
    near = np.abs(ref - CUTOFF) <= 30
    m1 = make_estimator(FINAL["model"], FEATS, FINAL["resampling"], CFG, y_te, FINAL_MONO).fit(X_te, y_te)
    cfg2 = copy.deepcopy(CFG)
    cfg2["seed"] = 999
    m2 = make_estimator(FINAL["model"], FEATS, FINAL["resampling"], cfg2, y_te, FINAL_MONO).fit(X_te, y_te)
    pure = float(np.abs(m1.predict_proba(X_te)[:, 1] - m2.predict_proba(X_te)[:, 1]).max())

    def evaluate(params, label):
        models = [train_variant(tv, s, params) for s in STABILITY_SEEDS]
        A = np.vstack([scores(m, X_te) for m in models]) >= CUTOFF
        flips = A.any(axis=0) != A.all(axis=0)
        path_res = {}
        for kind, p in paths.items():
            ok = np.vstack([scores(m, p["X"]) >= CUTOFF for m in models])
            path_res[kind] = {"paths": int(len(p["X"])), "deployed_score_median": float(np.median(p["deployed"])),
                              "still_approved_per_model": [round(float(r.mean()), 4) for r in ok],
                              "still_approved_all_3": round(float(ok.all(axis=0).mean()), 4),
                              "still_approved_all_3_thin": round(float(ok.all(axis=0)[p["thin"]].mean()), 4)}
        r = {"variant": label, "near_cutoff_rows": int(near.sum()), "near_cutoff_share": round(float(near.mean()), 4),
             "flip_share_near_cutoff": round(float(flips[near].mean()), 4), "flip_share_all": round(float(flips.mean()), 4),
             "test_auc_per_model": [round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 4) for m in models],
             "path_robustness": path_res}
        print(f"[stability] {label}: flips near cutoff {r['flip_share_near_cutoff']}, "
              f"min paths 495 all 3: {path_res['target495_minimal']['still_approved_all_3']}")
        return r

    out = {"seeds": STABILITY_SEEDS, "pure_seed_max_pd_diff": pure, "default": evaluate(None, "300 trees, lr 0.05")}
    if out["default"]["flip_share_near_cutoff"] > 0.10:
        out["slow"] = evaluate({"n_estimators": 1000, "learning_rate": 0.02}, "1000 trees, lr 0.02")
    return out


# =====================================================================
# 4. Sensitivity: mission correlation level and thin-filer definition (CV inside train)
# =====================================================================

def lift_cv(sim: pd.DataFrame) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["cv_repeats"] = 1
    labels = make_split(sim, cfg)
    train = sim[labels == "train"].reset_index(drop=True)
    folds = make_folds(train, cfg)
    pb, _ = cross_validate(train, folds, cfg, FINAL["model"], FEATS, FINAL["resampling"], FINAL_MONO)
    pg, _ = cross_validate(train, folds, cfg, FINAL["model"], FS["base"], FINAL["resampling"],
                           monotone_vector(FS["base"], tuple(FINAL["dropped_constraints"])))
    gain = pb[ev.KEY_THIN_AUC] - pg[ev.KEY_THIN_AUC]
    return {"cv_auc": round(float(pb[ev.KEY_AUC].mean()), 4), "thin_auc": round(float(pb[ev.KEY_THIN_AUC].mean()), 4),
            "thin_lift": round(float(gain.mean()), 4), "thin_lift_sd": round(float(gain.std()), 4),
            "thin_share": round(float(train[THIN].mean()), 4),
            "thin_default_rate": round(float(train.loc[train[THIN] == 1, TARGET].mean()), 4)}


def check_sensitivity(clean: pd.DataFrame, sim_main: pd.DataFrame) -> dict:
    base = add_thin_filer_flag(clean, SIM_CFG["thin_filer"]["rule"])
    out = {"mission_observed_r": {}, "thin_definition": {}}
    for r in (0.30, 0.32, 0.40, 0.50):
        c = copy.deepcopy(SIM_CFG)
        for n in MISSION:
            c["variables"][n]["target_corr"] = -r
        rec = {}
        try:
            sim, info = simulate(base, c)
            rec["within_0.60_cap"] = True
        except ValueError as err:                # cap or calibration infeasible: rerun with the cap lifted
            rec["within_0.60_cap"] = False
            rec["reason"] = str(err).splitlines()[0]
            c["max_inter_corr"] = 0.99
            # no shared factor (g = 0): the lowest inter-correlation reachable at this r
            c["shared_factor"]["target"] = {"stat": "max", "value": 0.0}
            sim, info = simulate(base, c)
        rec.update({"shared_factor": info["shared_factor"], "max_pair_corr": info["max_pair_corr"],
                    "observed_r": {n: info["corr_target"][n] for n in MISSION},
                    "auc_single": {n: info["auc_single"][n] for n in MISSION}, **lift_cv(sim)})
        out["mission_observed_r"][f"{r:.2f}"] = rec
        print(f"[sensitivity] r {r}: within cap {rec['within_0.60_cap']}, CV AUC {rec['cv_auc']}, lift {rec['thin_lift']}")
    for k in (1, 2, 3):
        rule = f"NumberOfOpenCreditLinesAndLoans <= {k} and NumberRealEstateLoansOrLines == 0"
        rec = {"rule": rule, **lift_cv(add_thin_filer_flag(sim_main.drop(columns=THIN), rule))}
        out["thin_definition"][f"<= {k} lines"] = rec
        print(f"[sensitivity] thin <= {k}: share {rec['thin_share']}, lift {rec['thin_lift']}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The official split replaces the earlier holdout re-selection; remove its old outputs
    for pattern in ("4_holdout*", "holdout_selection_*.csv", "5_sensitivity.json"):
        for old in OUT.glob(pattern):
            old.unlink()
    t0 = time.time()
    setup_mlflow(CFG)
    sim = pd.read_csv(ROOT / CFG["data"])
    labels = pd.read_csv(ROOT / CFG["outputs"]["split"]).set_index("row_id")["split"].loc[sim.index]
    tv, test = sim[labels.isin(["train", "validation"])], sim[labels == "test"]
    deployed = joblib.load(ROOT / SUMMARY["model_file"])
    clean = pd.read_csv(ROOT / load_yaml("data.yaml")["processed_path"])

    recourse, paths = check_recourse(test, deployed)
    save("1_recourse", recourse)
    log_audit_run("recourse", {"feasible_share": recourse["main_12m_495"]["feasible_share"],
                               "feasible_share_thin": recourse["main_12m_495"]["feasible_share_thin"]}, recourse)
    fairness = check_fairness(test)
    save("2_fairness", fairness)
    log_audit_run("fairness", {f"{k}_{m}": fairness[k][m] for k in ("gender", "age_band")
                               for m in ("di_ratio_min_over_max", "eo_tpr_diff", "eo_fpr_diff")}, fairness)
    stability = check_stability(tv, test, deployed, paths)
    save("3_stability", stability)
    log_audit_run("stability", {"flip_share_near_cutoff": stability["default"]["flip_share_near_cutoff"],
                                "paths495_min_all3": stability["default"]["path_robustness"]["target495_minimal"]["still_approved_all_3"]},
                  stability)
    sensitivity = check_sensitivity(clean, sim)
    save("4_sensitivity", sensitivity)
    log_audit_run("sensitivity", {f"lift_r{k}": v["thin_lift"] for k, v in sensitivity["mission_observed_r"].items()},
                  sensitivity)
    print(f"Audit done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
