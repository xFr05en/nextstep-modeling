"""Audit of the Step 5 model: recourse, fairness, stability, holdout, sensitivity.

Changes nothing: configs, the versioned final model file and the training MLflow runs stay as they are.
Outputs go to reports/audit/ and the MLflow experiment `nextstep-audit`.

Run from the project root:  python notebooks/audit.py   (about 6 minutes)
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.features import add_thin_filer_flag, feature_sets  # noqa: E402
from src.scoring import load_config as load_scoring, pd_to_score, score_frame  # noqa: E402
from src.data.simulator import load_config as load_sim, simulate  # noqa: E402
from src.train import (cross_validate, load_yaml, make_estimator, make_folds, monotone_vector,  # noqa: E402
                       pick_winner, setup_mlflow, stage_compare, stage_monotonic)

OUT = ROOT / "reports" / "audit"
CFG = load_yaml("train.yaml")
ACT = load_yaml("actionability.yaml")["features"]
SCORING = load_scoring()
CUTOFF = min(g["min_score"] for g in SCORING["grades"] if g["grade"] in SCORING["approval"]["approve_grades"])
FS = feature_sets()
FS = {"gmsc": FS["base"], "alt": FS["alt"], "both": FS["all"]}
FEATS = FS["both"]
TARGET, THIN = CFG["target"], CFG["thin_col"]
FINAL = json.loads((ROOT / "reports" / "final_summary.json").read_text())["winner"]
FINAL_MONO = monotone_vector(FEATS, tuple(FINAL["dropped_constraints"]))
HOLDOUT_SEED = 2026
STABILITY_SEEDS = [11, 22, 33]


def save(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def scores(est, X: pd.DataFrame) -> np.ndarray:
    return pd_to_score(est.predict_proba(X)[:, 1], SCORING)


# =====================================================================
# 1. Recourse feasibility
# =====================================================================

def vary_specs(extra: tuple[str, ...] = ()) -> dict:
    """Variables DiCE may vary (dice_vary, plus `extra`), with direction, step, bounds, months per step."""
    specs = {}
    for f in FEATS:
        e = ACT[f]
        if not (e["dice_vary"] or f in extra):
            continue
        step = e["step"]
        mult = isinstance(step, str) and step.endswith("%")
        specs[f] = {
            "dir": 1 if e["direction"] == "increase" else -1,
            "mult": mult,
            "step": float(step.rstrip("%")) / 100 if mult else float(step),
            "lo": e["bounds"][0], "hi": e["bounds"][1],
            "mps": float(e["months_per_step"]), "class": e["actionability"],
        }
    return specs


def apply_steps(x: np.ndarray, spec: dict, n) -> np.ndarray:
    """Value after n steps in the allowed direction, clipped to bounds. NaN stays NaN (no value to change)."""
    n = np.asarray(n, float)
    if spec["mult"]:
        y = x * (1 + spec["step"]) ** n
    else:
        y = x + spec["dir"] * spec["step"] * n
    lo = -np.inf if spec["lo"] is None else spec["lo"]
    hi = np.inf if spec["hi"] is None else spec["hi"]
    # never push a value the wrong way (e.g. utilization already below lower bound stays put)
    y = np.clip(y, lo, hi)
    return np.where(spec["dir"] > 0, np.maximum(y, x), np.minimum(y, x))


def corner_scores(est, X: pd.DataFrame, specs: dict, horizon: int, subsets) -> dict:
    out = {}
    for sub in subsets:
        Xm = X.copy()
        for v in sub:
            s = specs[v]
            Xm[v] = apply_steps(X[v].to_numpy(), s, np.floor(horizon / s["mps"] + 1e-9))
        out[sub] = scores(est, Xm)
    return out


def minimal_paths(est, X: pd.DataFrame, chosen: list, specs: dict, horizon: int, target: int,
                  iters: int = 14) -> pd.DataFrame:
    """Smallest uniform fraction t of the max change (per chosen subset) that reaches `target`."""
    lo, hi = np.zeros(len(X)), np.ones(len(X))

    def build(t):
        Xm = X.copy()
        for v, s in specs.items():
            use = np.array([v in c for c in chosen])
            if not use.any():
                continue
            n = np.ceil(t * np.floor(horizon / s["mps"] + 1e-9))
            new = apply_steps(X[v].to_numpy(), s, n)
            Xm.loc[use, v] = new[use]
        return Xm

    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = scores(est, build(mid)) >= target
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return build(hi)


def recourse_view(est, X_rej, thin_rej, specs, horizon, label, target):
    names = list(specs)
    subsets = [c for k in (1, 2, 3) for c in itertools.combinations(names, k)]
    sc = corner_scores(est, X_rej, specs, horizon, subsets + [tuple(names)])
    all_vars = sc.pop(tuple(names)) >= target
    best_k = np.full(len(X_rej), 0)
    chosen: list = [None] * len(X_rej)
    best_score = np.full(len(X_rej), -1)
    for k in (1, 2, 3):
        open_rows = best_k == 0
        for sub in [s for s in subsets if len(s) == k]:
            hit = (sc[sub] >= target) & open_rows
            better = hit & (sc[sub] > best_score)
            for i in np.where(better)[0]:
                chosen[i] = sub
            best_score = np.where(better, sc[sub], best_score)
        best_k = np.where(open_rows & (best_score >= target), k, best_k)
    feasible = best_k > 0
    top = pd.Series([" + ".join(c) for c in chosen if c]).value_counts().head(5)
    res = {
        "view": label, "horizon_months": horizon, "target_score": target, "variables": names,
        "rejected": int(len(X_rej)),
        "feasible_share": round(float(feasible.mean()), 4),
        "feasible_share_thin": round(float(feasible[thin_rej].mean()), 4),
        "feasible_share_not_thin": round(float(feasible[~thin_rej].mean()), 4),
        "not_feasible": int((~feasible).sum()),
        "vars_needed": {str(k): int((best_k == k).sum()) for k in (1, 2, 3)},
        "blocked_by_3_var_limit": round(float((~feasible & all_vars).mean()), 4),
        "blocked_even_with_all_vars": round(float((~feasible & ~all_vars).mean()), 4),
        "top_paths": {k: int(v) for k, v in top.items()},
    }
    return res, feasible, chosen, all_vars


def blocker_profile(X_rej: pd.DataFrame, score_rej, blocked: np.ndarray, specs, horizon, target) -> dict:
    sub = X_rej[blocked]
    pdc = load_yaml("data.yaml")["pastdue_columns"]
    serious = (sub[pdc[1]].fillna(0) > 0) | (sub[pdc[2]].fillna(0) > 0)
    no_room = np.zeros(len(sub), int)
    for v, s in specs.items():
        x = sub[v].to_numpy()
        no_room += (np.isnan(x) | (apply_steps(x, s, np.floor(horizon / s["mps"])) == x)).astype(int)
    return {
        "rows": int(blocked.sum()),
        "median_score": float(np.median(score_rej[blocked])),
        "median_points_short": float(np.median(target - score_rej[blocked])),
        "share_serious_late_60plus": round(float(serious.mean()), 4),
        "share_any_late": round(float((sub[pdc].fillna(0).sum(axis=1) > 0).mean()), 4),
        "share_special_code_96_98": round(float(sub["pastdue_special_code"].mean()), 4),
        "share_age_under_30": round(float((sub["age"] < 30).mean()), 4),
        "median_vars_with_no_room": float(np.median(no_room)),
        "share_income_missing": round(float(sub["income_missing"].mean()), 4),
    }


def build_paths(est, X_rej, thin_rej, feas, chosen, specs, target) -> dict:
    """Corner (max change) and minimal (just reaches target) paths for feasible applicants."""
    idx = np.where(feas)[0]
    Xf = X_rej.iloc[idx].reset_index(drop=True)
    ch = [chosen[i] for i in idx]
    X_corner = Xf.copy()
    for i, sub in enumerate(ch):
        for v in sub:
            s = specs[v]
            X_corner.loc[i, v] = apply_steps(np.array([Xf.loc[i, v]]), s, np.floor(12 / s["mps"]))[0]
    X_min = minimal_paths(est, Xf, ch, specs, 12, target)
    thin = thin_rej[idx]
    return {"corner": {"X": X_corner, "thin": thin, "deployed": scores(est, X_corner)},
            "minimal": {"X": X_min, "thin": thin, "deployed": scores(est, X_min)}}


def check_recourse(df: pd.DataFrame, est) -> tuple[dict, dict]:
    X = df[FEATS].astype(float)
    s0 = scores(est, X)
    rej = s0 < CUTOFF
    X_rej, thin_rej, s_rej = X[rej].reset_index(drop=True), df.loc[rej, THIN].to_numpy() == 1, s0[rej]
    target = SCORING["recourse_target_score"]
    # Original audit views: the 8 variables reviewed at the time (autopay included explicitly,
    # since it is no longer dice_vary). Decided view: current YAML (no autopay) and target 495.
    specs8 = vary_specs(extra=("autopay_ratio",))
    specs_now = vary_specs()
    no_time = {k: v for k, v in specs8.items() if v["class"] != "TIME_ONLY"}
    no_autopay = {k: v for k, v in specs8.items() if k != "autopay_ratio"}

    results, paths = {}, {}
    for label, sp, h, tgt in [("main_12m", specs8, 12, CUTOFF), ("all_24m", specs8, 24, CUTOFF),
                              ("no_time_only_12m", no_time, 12, CUTOFF), ("no_autopay_12m", no_autopay, 12, CUTOFF),
                              ("decided_12m_495", specs_now, 12, target)]:
        res, feas, chosen, all_vars = recourse_view(est, X_rej, thin_rej, sp, h, label, tgt)
        if (~feas).any():
            res["not_feasible_profile"] = blocker_profile(X_rej, s_rej, ~feas, sp, h, tgt)
        if label in ("main_12m", "decided_12m_495"):
            prefix = "original_475" if label == "main_12m" else "decided_495"
            for kind, p in build_paths(est, X_rej, thin_rej, feas, chosen, sp, tgt).items():
                paths[f"{prefix}_{kind}"] = p
            res["minimal_path_reaches_target"] = round(float((paths[f"{prefix}_minimal"]["deployed"] >= tgt).mean()), 4)
        results[label] = res
        print(f"[recourse] {label}: feasible {res['feasible_share']:.4f} (thin {res['feasible_share_thin']:.4f})")

    results["rejected_by_deployed_model"] = int(rej.sum())
    results["rejected_share"] = round(float(rej.mean()), 4)
    results["rejected_share_thin"] = round(float(rej[df[THIN].to_numpy() == 1].mean()), 4)
    return results, paths


# =====================================================================
# 2. Fairness by age group (out-of-fold decisions of the current setup)
# =====================================================================

def check_fairness(df: pd.DataFrame) -> dict:
    oof = pd.read_csv(ROOT / CFG["outputs"]["oof"])
    assert len(oof) == len(df) and (oof["default"].to_numpy() == df[TARGET].to_numpy()).all()
    groups = {
        "age": pd.cut(df["age"], [0, 29, 49, 200], labels=["under_30", "30_to_49", "50_plus"]),
        "gender": df["gender_female"].map({0: "male", 1: "female"}),
    }
    out = {}
    for gname, g in groups.items():
        rows = {}
        for lvl in g.cat.categories if hasattr(g, "cat") else sorted(g.unique()):
            m = (g == lvl).to_numpy()
            a, y = oof["approved"].to_numpy()[m], oof["default"].to_numpy()[m]
            p = a.mean()
            rows[str(lvl)] = {
                "n": int(m.sum()), "default_rate": round(float(y.mean()), 4),
                "approval_rate": round(float(p), 4),
                "approval_ci95": [round(float(p - 1.96 * np.sqrt(p * (1 - p) / m.sum())), 4),
                                  round(float(p + 1.96 * np.sqrt(p * (1 - p) / m.sum())), 4)],
                "approval_if_repaid": round(float(a[y == 0].mean()), 4),
                "approval_if_defaulted": round(float(a[y == 1].mean()), 4),
            }
        best = max(r["approval_rate"] for r in rows.values())
        for r in rows.values():
            r["di_ratio"] = round(r["approval_rate"] / best, 4)
        repaid = [r["approval_if_repaid"] for r in rows.values()]
        defaulted = [r["approval_if_defaulted"] for r in rows.values()]
        out[gname] = {
            "groups": rows,
            "min_di_ratio": min(r["di_ratio"] for r in rows.values()),
            "eo_gap_repaid": round(max(repaid) - min(repaid), 4),
            "eo_gap_defaulted": round(max(defaulted) - min(defaulted), 4),
            "eo_gap": round(max(max(repaid) - min(repaid), max(defaulted) - min(defaulted)), 4),
        }
        print(f"[fairness] {gname}: min DI {out[gname]['min_di_ratio']}, EO gap {out[gname]['eo_gap']}")
    return out


# =====================================================================
# 4. Holdout (full selection rerun on 80%) and 3. Stability
# =====================================================================

def boot_ci(fn, n: int, B: int = 500, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def check_holdout(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    strata = df[TARGET] * 2 + df[THIN]
    tr_idx, ho_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=strata, random_state=HOLDOUT_SEED)
    train, hold = df.iloc[tr_idx].reset_index(drop=True), df.iloc[ho_idx].reset_index(drop=True)

    cfg = copy.deepcopy(CFG)
    cfg["mlflow"]["experiments"] = {k: "nextstep-audit" for k in cfg["mlflow"]["experiments"]}
    cfg["outputs"]["comparison"] = "reports/audit/holdout_selection_comparison.csv"
    cfg["outputs"]["monotonic"] = "reports/audit/holdout_selection_monotonic.csv"
    setup_mlflow(cfg)
    folds = make_folds(train, cfg)
    t0 = time.time()
    comparison = stage_compare(train, folds, cfg, FS)
    mono = stage_monotonic(train, folds, cfg, FS, comparison)
    winner = pick_winner(comparison[comparison["features"] == "both"], cfg)
    name, res = winner["model"], winner["resampling"]
    sel = mono[(mono["model"] == name) & (mono["selected"] == True)] if name in cfg["monotonic"]["models"] else None  # noqa: E712
    dropped = () if sel is None or sel.iloc[0]["dropped"] == "none" else tuple(sel.iloc[0]["dropped"].split(","))
    m_vec = monotone_vector(FEATS, dropped) if sel is not None else None
    calibrate = res != "none"
    print(f"[holdout] selection on 80% took {time.time() - t0:.0f}s, winner {name}/{res}, dropped {dropped}")

    y_tr, y_ho = train[TARGET].to_numpy(), hold[TARGET].to_numpy()
    thin_ho = hold[THIN].to_numpy() == 1
    est = make_estimator(name, FEATS, res, cfg, y_tr, m_vec, calibrate).fit(train[FEATS].astype(float), y_tr)
    g_mono = monotone_vector(FS["gmsc"], dropped) if m_vec else None
    est_g = make_estimator(name, FS["gmsc"], res, cfg, y_tr, g_mono, calibrate).fit(train[FS["gmsc"]].astype(float), y_tr)
    p_ho = est.predict_proba(hold[FEATS].astype(float))[:, 1]
    p_tr = est.predict_proba(train[FEATS].astype(float))[:, 1]
    p_g = est_g.predict_proba(hold[FS["gmsc"]].astype(float))[:, 1]

    def thin_auc(p, idx):
        t = thin_ho[idx]
        return roc_auc_score(y_ho[idx][t], p[idx][t])

    sc = score_frame(p_ho, SCORING)
    sc["default"], sc["thin"] = y_ho, thin_ho
    grades = sc.groupby("grade").agg(share=("pd", "size"), mean_pd=("pd", "mean"), default_rate=("default", "mean"))
    grades["share"] = grades["share"] / len(sc)
    n = len(hold)
    result = {
        "holdout_rows": n, "train_rows": len(train), "split_seed": HOLDOUT_SEED,
        "winner_on_80pct": {"model": name, "resampling": res, "dropped_constraints": list(dropped),
                            "same_as_step5": name == FINAL["model"] and res == FINAL["resampling"]
                            and list(dropped) == FINAL["dropped_constraints"]},
        "auc": round(roc_auc_score(y_ho, p_ho), 4),
        "auc_ci95": boot_ci(lambda i: roc_auc_score(y_ho[i], p_ho[i]), n),
        "ks": round(ev.ks_stat(y_ho, p_ho), 4),
        "ks_ci95": boot_ci(lambda i: ev.ks_stat(y_ho[i], p_ho[i]), n),
        "thin_auc": round(roc_auc_score(y_ho[thin_ho], p_ho[thin_ho]), 4),
        "thin_auc_gmsc_only": round(roc_auc_score(y_ho[thin_ho], p_g[thin_ho]), 4),
        "thin_auc_gain": round(roc_auc_score(y_ho[thin_ho], p_ho[thin_ho]) - roc_auc_score(y_ho[thin_ho], p_g[thin_ho]), 4),
        "thin_auc_gain_ci95": boot_ci(lambda i: thin_auc(p_ho, i) - thin_auc(p_g, i), n),
        "psi_train_vs_holdout": round(ev.psi(p_tr, p_ho), 5),
        "brier": round(brier_score_loss(y_ho, p_ho), 5),
        "mean_pd": round(float(p_ho.mean()), 4), "default_rate": round(float(y_ho.mean()), 4),
        "approval_rate": round(float(sc["approved"].mean()), 4),
        "approval_rate_thin": round(float(sc.loc[sc.thin, "approved"].mean()), 4),
        "grades": grades.round(4).to_dict(orient="index"),
    }
    print(f"[holdout] AUC {result['auc']} KS {result['ks']} thin gain {result['thin_auc_gain']}")
    setup = {"name": FINAL["model"], "res": FINAL["resampling"], "mono": FINAL_MONO,
             "calibrate": FINAL["resampling"] != "none"}
    return result, train, hold, setup


def train_variant(train: pd.DataFrame, setup: dict, seed: int, params: dict | None = None):
    """Final setup trained on a seed-specific random 80% subsample of the training rows."""
    sub = train.sample(frac=0.8, random_state=seed)
    cfg = copy.deepcopy(CFG)
    if params:
        cfg["models"][setup["name"]].update(params)
    y = sub[TARGET].to_numpy()
    return make_estimator(setup["name"], FEATS, setup["res"], cfg, y, setup["mono"], setup["calibrate"]).fit(
        sub[FEATS].astype(float), y)


def check_stability(train, hold, setup, paths) -> dict:
    X_ho, y_ho = hold[FEATS].astype(float), hold[TARGET].to_numpy()
    cfg_ref = copy.deepcopy(CFG)
    ref = make_estimator(setup["name"], FEATS, setup["res"], cfg_ref, train[TARGET].to_numpy(), setup["mono"],
                         setup["calibrate"]).fit(train[FEATS].astype(float), train[TARGET].to_numpy())
    ref_score = scores(ref, X_ho)
    near = np.abs(ref_score - CUTOFF) <= 30

    # Pure seed check: with no row/column subsampling, random_state should not matter
    cfg_seed = copy.deepcopy(CFG)
    m1 = make_estimator(setup["name"], FEATS, setup["res"], cfg_seed, y_ho, setup["mono"]).fit(X_ho, y_ho)
    cfg_seed["seed"] = 999
    m2 = make_estimator(setup["name"], FEATS, setup["res"], cfg_seed, y_ho, setup["mono"]).fit(X_ho, y_ho)
    pure_seed_diff = float(np.abs(m1.predict_proba(X_ho)[:, 1] - m2.predict_proba(X_ho)[:, 1]).max())

    def evaluate(variant_params, label):
        models = [train_variant(train, setup, s, variant_params) for s in STABILITY_SEEDS]
        S = np.vstack([scores(m, X_ho) for m in models])
        A = S >= CUTOFF
        flips = A.any(axis=0) != A.all(axis=0)
        path_res = {}
        for kind, p in paths.items():
            # "still approved" always means score >= approval cutoff (475) under a retrained model
            ok = np.vstack([scores(m, p["X"]) >= CUTOFF for m in models])
            path_res[kind] = {
                "paths": int(len(p["X"])),
                "deployed_score_median": float(np.median(p["deployed"])),
                "still_approved_per_model": [round(float(r.mean()), 4) for r in ok],
                "still_approved_mean": round(float(ok.mean()), 4),
                "still_approved_all_3": round(float(ok.all(axis=0).mean()), 4),
                "still_approved_all_3_thin": round(float(ok.all(axis=0)[p["thin"]].mean()), 4),
            }
        # Margin estimate (original minimal paths): a path aimed at cutoff + m stays approved if the
        # score drop under a retrained model is at most (deployed score - cutoff) + m
        p = paths["original_475_minimal"]
        drop = p["deployed"][None, :] - np.vstack([scores(m, p["X"]) for m in models])
        margin = {}
        for m in (0, 10, 20, 30, 40):
            ok = drop <= (p["deployed"] - CUTOFF)[None, :] + m
            margin[str(m)] = {"per_model_mean": round(float(ok.mean()), 4), "all_3": round(float(ok.all(axis=0).mean()), 4)}
        path_res["original_minimal_margin_estimate"] = margin
        r = {
            "variant": label, "seeds": STABILITY_SEEDS,
            "near_cutoff_rows": int(near.sum()), "near_cutoff_share_of_holdout": round(float(near.mean()), 4),
            "flip_share_near_cutoff": round(float(flips[near].mean()), 4),
            "flip_share_all": round(float(flips.mean()), 4),
            "auc_per_model": [round(roc_auc_score(y_ho, m.predict_proba(X_ho)[:, 1]), 4) for m in models],
            "path_robustness": path_res,
        }
        print(f"[stability] {label}: flips near cutoff {r['flip_share_near_cutoff']:.3f}, "
              f"minimal paths still approved (all 3): original {path_res['original_475_minimal']['still_approved_all_3']:.3f}, "
              f"decided 495 {path_res['decided_495_minimal']['still_approved_all_3']:.3f}")
        return r

    out = {"pure_seed_max_pd_diff": pure_seed_diff, "default": evaluate(None, "300 trees, lr 0.05")}
    if out["default"]["flip_share_near_cutoff"] > 0.10:
        out["slow"] = evaluate({"n_estimators": 1000, "learning_rate": 0.02}, "1000 trees, lr 0.02")
    return out


# =====================================================================
# 5. Sensitivity
# =====================================================================

def gain_cv(sim: pd.DataFrame, cfg: dict) -> dict:
    folds = make_folds(sim, cfg)
    pf_b, _ = cross_validate(sim, folds, cfg, FINAL["model"], FEATS, FINAL["resampling"], FINAL_MONO)
    pf_g, _ = cross_validate(sim, folds, cfg, FINAL["model"], FS["gmsc"], FINAL["resampling"],
                             monotone_vector(FS["gmsc"], tuple(FINAL["dropped_constraints"])))
    gain = pf_b[ev.KEY_THIN_AUC] - pf_g[ev.KEY_THIN_AUC]
    return {"thin_auc_gain": round(float(gain.mean()), 4), "thin_auc_gain_std": round(float(gain.std()), 4),
            "auc": round(float(pf_b[ev.KEY_AUC].mean()), 4), "thin_auc": round(float(pf_b[ev.KEY_THIN_AUC].mean()), 4),
            "thin_share": round(float(sim[THIN].mean()), 4),
            "thin_default_rate": round(float(sim.loc[sim[THIN] == 1, TARGET].mean()), 4)}


def check_sensitivity(clean: pd.DataFrame, sim_main: pd.DataFrame) -> dict:
    scfg = load_sim()
    out = {"latent_r": {}, "thin_definition": {}}
    for r in (0.30, 0.35, 0.50):
        c = copy.deepcopy(scfg)
        for v in c["variables"].values():
            v["target_corr"] = -r
        sim, info = simulate(add_thin_filer_flag(clean, scfg["thin_filer"]["rule"]), c)
        res = gain_cv(sim, CFG)
        res["noise_rounds"] = info["noise_rounds"]
        res["observed_pearson_range"] = [min(info["corr_target"].values()), max(info["corr_target"].values())]
        out["latent_r"][str(r)] = res
        print(f"[sensitivity] latent r {r}: thin gain {res['thin_auc_gain']}")
    for k in (1, 2, 3):
        rule = f"NumberOfOpenCreditLinesAndLoans <= {k} and NumberRealEstateLoansOrLines == 0"
        sim = add_thin_filer_flag(sim_main.drop(columns=THIN), rule)
        res = gain_cv(sim, CFG)
        res["rule"] = rule
        out["thin_definition"][f"<= {k} lines"] = res
        print(f"[sensitivity] thin <= {k}: share {res['thin_share']}, gain {res['thin_auc_gain']}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim = pd.read_csv(ROOT / CFG["data"])
    clean = pd.read_csv(ROOT / load_yaml("data.yaml")["processed_path"])
    deployed = joblib.load(ROOT / json.loads((ROOT / CFG["outputs"]["final_summary"]).read_text())["model_file"])

    recourse, paths = check_recourse(sim, deployed)
    save("1_recourse", recourse)
    save("2_fairness", check_fairness(sim))
    holdout, train, hold, setup = check_holdout(sim)
    save("4_holdout", holdout)
    save("3_stability", check_stability(train, hold, setup, paths))
    save("5_sensitivity", check_sensitivity(clean, sim))
    print(f"Audit done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
