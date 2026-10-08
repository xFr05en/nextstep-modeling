"""Model comparison, monotonic constraints and final model (Step 5).

Run from the project root:
  python -m src.train --stage compare     # 3 models x 3 feature sets x 3 resampling options
  python -m src.train --stage monotonic   # constrained vs unconstrained boosting models
  python -m src.train --stage final       # winner + constraints (+ Platt if resampled), scores, refit
  python -m src.train --stage all

Leakage rule: imputation, LR transforms, scaling, SMOTE and calibration are steps of one
pipeline, so they are fit on the training fold only and merely applied to the held-out fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import skops.io  # noqa: E402
import yaml  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src import evaluate as ev  # noqa: E402
from src.features import feature_sets  # noqa: E402
from src.scoring import load_config as load_scoring, score_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_KEYS = [ev.KEY_AUC, ev.KEY_KS, ev.KEY_THIN_AUC, ev.KEY_PSI, ev.KEY_BRIER,
                ev.KEY_PRECISION, ev.KEY_RECALL, ev.KEY_F1]


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Pipeline pieces ----

class LogCap(BaseEstimator, TransformerMixin):
    """log1p (values below 0 clipped to 0), then cap at a quantile learned on the training fold."""

    def __init__(self, cols=(), q: float = 0.99):
        self.cols = cols
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.caps_ = {c: np.quantile(np.log1p(np.clip(X[:, c], 0, None)), self.q) for c in self.cols}
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        for c, cap in self.caps_.items():
            X[:, c] = np.minimum(np.log1p(np.clip(X[:, c], 0, None)), cap)
        return X


def make_model(name: str, resampling: str, monotone: list[int] | None, cfg: dict):
    params, seed = cfg["models"][name], cfg["seed"]
    balanced = "balanced" if resampling == "class_weight" else None
    if name == "lr":
        return LogisticRegression(**params, class_weight=balanced)
    if name == "lgbm":
        extra = {"monotone_constraints": monotone} if monotone else {}
        return LGBMClassifier(**params, class_weight=balanced, random_state=seed, verbose=-1, n_jobs=-1, **extra)
    if name == "xgb":
        extra = {"monotone_constraints": tuple(monotone)} if monotone else {}
        return XGBClassifier(**params, random_state=seed, n_jobs=-1, eval_metric="logloss", **extra)
    raise ValueError(name)


def build_pipeline(name: str, features: list[str], resampling: str, cfg: dict,
                   monotone: list[int] | None = None) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if name == "lr":
        idx = [features.index(c) for c in cfg["lr_log_cap_columns"] if c in features]
        steps.append(("logcap", LogCap(cols=idx, q=cfg["lr_cap_quantile"])))
    if name == "lr" or resampling == "smote":
        steps.append(("scale", StandardScaler()))  # SMOTE uses distances, so scale first
    if resampling == "smote":
        steps.append(("smote", SMOTE(random_state=cfg["seed"])))
    steps.append(("model", make_model(name, resampling, monotone, cfg)))
    return Pipeline(steps)


def make_estimator(name, features, resampling, cfg, y_train, monotone=None, calibrate=False):
    pipe = build_pipeline(name, features, resampling, cfg, monotone)
    if name == "xgb" and resampling == "class_weight":
        pipe.set_params(model__scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()))
    if calibrate:
        c = cfg["calibration"]
        return CalibratedClassifierCV(pipe, method=c["method"], cv=c["cv"], ensemble=False)
    return pipe


def monotone_vector(features: list[str], dropped: tuple[str, ...] = ()) -> list[int]:
    feats = load_yaml("actionability.yaml")["features"]
    return [0 if f in dropped else int(feats[f]["monotone"]) for f in features]


# ---- Cross-validation ----

def make_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Stratified (stratify=y) train / validation / test labels, e.g. 70:15:15. Index = df.index."""
    sp = cfg["split"]
    y = df[cfg["target"]]
    rest, test = train_test_split(df.index, test_size=sp["test"], stratify=y, random_state=sp["seed"])
    val_share = sp["validation"] / (sp["train"] + sp["validation"])
    train, val = train_test_split(rest, test_size=val_share, stratify=y.loc[rest], random_state=sp["seed"])
    labels = pd.Series("train", index=df.index, name="split")
    labels.loc[val] = "validation"
    labels.loc[test] = "test"
    return labels


def save_split(labels: pd.Series, cfg: dict) -> None:
    out = ROOT / cfg["outputs"]["split"]
    pd.DataFrame({"row_id": labels.index, "split": labels.to_numpy()}).to_csv(out, index=False)


def make_folds(df: pd.DataFrame, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    strata = df[cfg["target"]].to_numpy() * 2 + df[cfg["thin_col"]].to_numpy()
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    return list(skf.split(df, strata))


def cross_validate(df, folds, cfg, name, features, resampling, monotone=None, calibrate=False):
    """Returns per-fold metrics (DataFrame) and out-of-fold PD predictions."""
    X = df[features].astype(float)
    y = df[cfg["target"]].to_numpy()
    thin = df[cfg["thin_col"]].to_numpy() == 1
    oof = np.zeros(len(df))
    rows = []
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        est = make_estimator(name, features, resampling, cfg, y[tr], monotone, calibrate)
        est.fit(X.iloc[tr], y[tr])
        p_tr = est.predict_proba(X.iloc[tr])[:, 1]
        p_te = est.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p_te
        rows.append({"fold": k, **ev.fold_metrics(y[te], p_te, p_tr, thin[te])})
    per_fold = pd.DataFrame(rows)
    per_fold.attrs["seconds"] = round(time.time() - t0, 1)
    return per_fold, oof


def summarize(per_fold: pd.DataFrame) -> dict:
    out = {}
    for k in SUMMARY_KEYS:
        out[f"{k}_mean"] = float(per_fold[k].mean())
        out[f"{k}_std"] = float(per_fold[k].std())
    return out


# ---- MLflow ----

def setup_mlflow(cfg: dict) -> None:
    d = ROOT / cfg["mlflow"]["tracking_dir"]
    d.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{d / 'mlflow.db'}")


def use_experiment(cfg: dict, stage: str) -> None:
    name = cfg["mlflow"]["experiments"][stage]
    if mlflow.get_experiment_by_name(name) is None:
        art = ROOT / cfg["mlflow"]["tracking_dir"] / "artifacts" / name
        mlflow.create_experiment(name, artifact_location=art.as_uri())
    mlflow.set_experiment(name)


def data_md5(cfg: dict) -> str:
    return hashlib.md5((ROOT / cfg["data"]).read_bytes()).hexdigest()[:8]


def log_run(cfg, stage, run_name, params, per_fold, extra_metrics, features, monotone=None):
    use_experiment(cfg, stage)
    sim_b = load_yaml("simulator.yaml")["credit_link"]["b"]
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"step": "5", "stage": stage, "owner": "wonbin"})
        mlflow.log_params({**params, "seed": cfg["seed"], "n_splits": cfg["n_splits"],
                           "sim_b": sim_b, "data_md5": data_md5(cfg),
                           **{f"hp_{k}": v for k, v in cfg["models"][params["model"]].items()}})
        for _, r in per_fold.iterrows():
            for k in SUMMARY_KEYS:
                mlflow.log_metric(k, float(r[k]), step=int(r["fold"]))
        mlflow.log_metrics({**summarize(per_fold), **extra_metrics})
        mlflow.log_text(per_fold.to_csv(index=False), "per_fold_metrics.csv")
        mlflow.log_dict({"features": features}, "features.json")
        if monotone is not None:
            mlflow.log_dict(dict(zip(features, monotone)), "monotone.json")


# ---- Stages ----

def pick_winner(results: pd.DataFrame, cfg: dict) -> pd.Series:
    """Highest AUC; runs within 1 fold SD of the best are ties, broken by resampling preference."""
    best = results.loc[results["auc_mean"].idxmax()]
    ties = results[results["auc_mean"] >= best["auc_mean"] - best["auc_std"]].copy()
    rank = {r: i for i, r in enumerate(cfg["resampling_preference"])}
    ties["_pref"] = ties["resampling"].map(rank)
    return ties.sort_values(["_pref", "auc_mean"], ascending=[True, False]).iloc[0]


def stage_compare(df, folds, cfg, fs) -> pd.DataFrame:
    rows = []
    for name in cfg["model_names"]:
        for res in cfg["resampling"]:
            cv = {s: cross_validate(df, folds, cfg, name, fs[s], res) for s in cfg["feature_sets"]}
            gain = cv["both"][0][ev.KEY_THIN_AUC] - cv["gmsc"][0][ev.KEY_THIN_AUC]
            for s in cfg["feature_sets"]:
                per_fold = cv[s][0]
                extra = {}
                if s == "both":
                    extra = {f"{ev.KEY_THIN_GAIN}_mean": float(gain.mean()),
                             f"{ev.KEY_THIN_GAIN}_std": float(gain.std())}
                run = f"{name}__{s}__{res}"
                log_run(cfg, "compare", run, {"model": name, "feature_set": s, "resampling": res,
                                              "monotone_set": "none"}, per_fold, extra, fs[s])
                rows.append({"run_name": run, "model": name, "features": s, "resampling": res,
                             **summarize(per_fold), **extra, "seconds": per_fold.attrs["seconds"]})
                print(f"{run:32s} AUC {rows[-1]['auc_mean']:.4f}  KS {rows[-1]['ks_mean']:.4f}  "
                      f"thinAUC {rows[-1]['thin_auc_mean']:.4f}  {per_fold.attrs['seconds']}s")
    out = pd.DataFrame(rows)
    out.round(5).to_csv(ROOT / cfg["outputs"]["comparison"], index=False)
    return out


def stage_monotonic(train, val, cfg, fs, comparison: pd.DataFrame) -> pd.DataFrame:
    """Constraint cost measured on validation: unconstrained vs constrained, both fit on train."""
    feats = fs["both"]
    mc, bs = cfg["monotonic"], cfg["bootstrap"]
    X_tr, y_tr = train[feats].astype(float), train[cfg["target"]].to_numpy()
    X_va, y_va = val[feats].astype(float), val[cfg["target"]].to_numpy()
    rows = []
    for name in mc["models"]:
        cand = comparison[(comparison["model"] == name) & (comparison["features"] == "both")]
        res = pick_winner(cand, cfg)["resampling"]
        p_base = make_estimator(name, feats, res, cfg, y_tr).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
        auc_base = roc_auc_score(y_va, p_base)
        dropped: list[str] = []
        for step in range(len(mc["drop_order"]) + 1):
            mono = monotone_vector(feats, tuple(dropped))
            p_con = make_estimator(name, feats, res, cfg, y_tr, mono).fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
            auc_con = roc_auc_score(y_va, p_con)
            loss = auc_base - auc_con
            ci = ev.bootstrap_ci(lambda i: roc_auc_score(y_va[i], p_base[i]) - roc_auc_score(y_va[i], p_con[i]),
                                 len(y_va), bs["n"], bs["seed"])
            label = "full" if not dropped else "no-" + "-".join(d.lower()[:12] for d in dropped)
            use_experiment(cfg, "monotonic")
            with mlflow.start_run(run_name=f"{name}__both__{res}__mono-{label}"):
                mlflow.set_tags({"step": "5", "stage": "monotonic", "owner": "wonbin", "evaluated_on": "validation"})
                mlflow.log_params({"model": name, "feature_set": "both", "resampling": res, "monotone_set": label,
                                   "dropped_constraints": ",".join(dropped) or "none", "seed": cfg["seed"],
                                   "data_md5": data_md5(cfg)})
                mlflow.log_metrics({"val_auc_unconstrained": auc_base, "val_auc_constrained": auc_con,
                                    ev.KEY_MONO_LOSS: loss, f"{ev.KEY_MONO_LOSS}_ci_low": ci[0],
                                    f"{ev.KEY_MONO_LOSS}_ci_high": ci[1]})
                mlflow.log_dict(dict(zip(feats, mono)), "monotone.json")
            rows.append({"model": name, "resampling": res, "monotone_set": label,
                         "dropped": ",".join(dropped) or "none", "evaluated_on": "validation",
                         "auc_unconstrained": auc_base, "auc_constrained": auc_con,
                         ev.KEY_MONO_LOSS: loss, "loss_ci95_low": ci[0], "loss_ci95_high": ci[1],
                         "n_constrained": int(np.count_nonzero(mono))})
            print(f"{name} {res} mono-{label}: val AUC {auc_con:.4f} loss {loss:+.4f} CI {ci}")
            if loss <= mc["max_auc_loss"] or step == len(mc["drop_order"]):
                rows[-1]["selected"] = loss <= mc["max_auc_loss"]
                break
            dropped.append(mc["drop_order"][step])
    out = pd.DataFrame(rows)
    out.round(5).to_csv(ROOT / cfg["outputs"]["monotonic"], index=False)
    return out


def monotonic_violations(est, X: pd.DataFrame, features, monotone, n_rows=300, n_grid=15, seed=0) -> dict:
    """Move each constrained feature over its range for sample rows; count wrong-direction steps."""
    rng = np.random.default_rng(seed)
    sample = X.iloc[rng.choice(len(X), n_rows, replace=False)]
    out = {}
    for f, m in zip(features, monotone):
        if m == 0:
            continue
        grid = np.unique(np.nanquantile(X[f], np.linspace(0.01, 0.99, n_grid)))
        preds = []
        for v in grid:
            s = sample.copy()
            s[f] = v
            preds.append(est.predict_proba(s)[:, 1])
        steps = np.diff(np.vstack(preds), axis=0) * m  # should be >= 0
        out[f] = int((steps < -1e-9).sum())
    return out


def log_final_model(est, example: pd.DataFrame) -> list[str]:
    """Log the fitted model to the active MLflow run so pyfunc serves probabilities.

    MLflow saves sklearn models with skops, which only loads listed types. The model was trained
    in this process, so its own classes (pipeline, booster, etc.) are listed as trusted.
    pyfunc serves predict_proba (the default would be predict = class labels). Column 1 is the PD.
    """
    trusted = skops.io.get_untrusted_types(data=skops.io.dumps(est))
    mlflow.sklearn.log_model(est, name="model", input_example=example, skops_trusted_types=trusted,
                             pyfunc_predict_fn="predict_proba",
                             signature=infer_signature(example, est.predict_proba(example)))
    return trusted


def stage_final(train, val, test, cfg, fs, comparison, monotonic) -> dict:
    """Fit the winner on train+validation and evaluate it once on the test set.

    Thresholds are fixed rules from config/scoring.yaml; no test data is used to set anything."""
    winner = pick_winner(comparison[comparison["features"] == "both"], cfg)
    name, res = winner["model"], winner["resampling"]
    feats = fs["both"]
    dropped: tuple[str, ...] = ()
    mono = None
    sel = None
    if name in cfg["monotonic"]["models"]:
        sel = monotonic[(monotonic["model"] == name) & (monotonic["selected"] == True)]  # noqa: E712
        if sel.empty:
            raise ValueError(f"No constraint set for {name} stays within the AUC loss limit")
        d = sel.iloc[0]["dropped"]
        dropped = () if d == "none" else tuple(d.split(","))
        mono = monotone_vector(feats, dropped)
    # Calibration (only if resampled): CalibratedClassifierCV with internal CV, refit on train+validation
    calibrate = res != "none"
    target, thin_col = cfg["target"], cfg["thin_col"]
    tv = pd.concat([train, val])
    X_tv, y_tv = tv[feats].astype(float), tv[target].to_numpy()
    X_te, y_te = test[feats].astype(float), test[target].to_numpy()
    thin = test[thin_col].to_numpy() == 1

    final_est = make_estimator(name, feats, res, cfg, y_tv, mono, calibrate).fit(X_tv, y_tv)
    p_te = final_est.predict_proba(X_te)[:, 1]
    p_tv = final_est.predict_proba(X_tv)[:, 1]
    mono_g = monotone_vector(fs["gmsc"], dropped) if mono else None
    est_g = make_estimator(name, fs["gmsc"], res, cfg, y_tv, mono_g, calibrate).fit(tv[fs["gmsc"]].astype(float), y_tv)
    p_g = est_g.predict_proba(test[fs["gmsc"]].astype(float))[:, 1]

    scoring = load_scoring()
    bs = cfg["bootstrap"]
    n = len(y_te)

    def thin_auc(p, i):
        t = thin[i]
        return roc_auc_score(y_te[i][t], p[i][t])

    def ci(fn):
        return ev.bootstrap_ci(fn, n, bs["n"], bs["seed"])

    cls = ev.classification_metrics(y_te, p_te, scoring)
    test_metrics = {
        "auc": round(roc_auc_score(y_te, p_te), 4), "auc_ci95": ci(lambda i: roc_auc_score(y_te[i], p_te[i])),
        "ks": round(ev.ks_stat(y_te, p_te), 4), "ks_ci95": ci(lambda i: ev.ks_stat(y_te[i], p_te[i])),
        "thin_auc": round(thin_auc(p_te, np.arange(n)), 4),
        "thin_auc_gmsc_only": round(thin_auc(p_g, np.arange(n)), 4),
        "thin_auc_lift": round(thin_auc(p_te, np.arange(n)) - thin_auc(p_g, np.arange(n)), 4),
        # Paired bootstrap: same resampled test rows for both models, lift computed inside each resample
        "thin_auc_lift_ci95": ci(lambda i: thin_auc(p_te, i) - thin_auc(p_g, i)),
        "psi_trainval_vs_test": round(ev.psi(p_tv, p_te), 5),
        "brier": round(brier_score_loss(y_te, p_te), 5),
        **{k: round(float(v), 4) for k, v in cls.items()},
        "f1_ci95": ci(lambda i: ev.classification_metrics(y_te[i], p_te[i], scoring)[ev.KEY_F1]),
        "rows": int(n), "thin_rows": int(thin.sum()), "thin_defaults": int(y_te[thin].sum()),
    }
    mono_row = sel.iloc[0] if sel is not None else None
    mono_loss = None if mono_row is None else {
        "loss": round(float(mono_row[ev.KEY_MONO_LOSS]), 4),
        "ci95": [round(float(mono_row["loss_ci95_low"]), 4), round(float(mono_row["loss_ci95_high"]), 4)],
        "evaluated_on": "validation"}

    # Scores, grades, approval on the test set (the shipped model, never in-sample)
    sc = score_frame(p_te, scoring)
    sc.insert(0, "row_id", test.index.to_numpy())
    sc["default"], sc["thin_filer"] = y_te, thin.astype(int)
    grade_tbl = (sc.groupby("grade")
                 .agg(rows=("pd", "size"), mean_pd=("pd", "mean"), default_rate=("default", "mean"),
                      min_score=("score", "min"), max_score=("score", "max"))
                 .assign(share=lambda t: t["rows"] / len(sc),
                         thin_share=sc[sc.thin_filer == 1].groupby("grade").size() / sc.thin_filer.sum()))
    grade_tbl.round(4).to_csv(ROOT / cfg["outputs"]["grade_table"])
    sc.to_csv(ROOT / cfg["outputs"]["test_scores"], index=False)

    violations = monotonic_violations(final_est, X_tv, feats, mono) if mono else {}
    joblib.dump(final_est, ROOT / cfg["outputs"]["model"])

    summary = {
        "winner": {"model": name, "resampling": res, "calibrated": calibrate,
                   "dropped_constraints": list(dropped),
                   "monotone": dict(zip(feats, mono)) if mono else None},
        "trained_on": "train+validation", "evaluated_on": "test",
        "split_rows": {k: int(v) for k, v in pd.Series(["train"] * len(train) + ["validation"] * len(val)
                                                       + ["test"] * len(test)).value_counts().items()},
        "cv_on_train": {k: round(float(winner[k]), 4) for k in ("auc_mean", "auc_std", "ks_mean", "thin_auc_mean")},
        "test": test_metrics,
        "monotonic_loss_validation": mono_loss,
        "cutoff_rule": "approve if grade in approve_grades (score >= 475), fixed in config/scoring.yaml",
        "mean_pd_test": round(float(p_te.mean()), 4), "default_rate_test": round(float(y_te.mean()), 4),
        "approval_rate": round(float(sc["approved"].mean()), 4),
        "approval_rate_thin": round(float(sc.loc[sc.thin_filer == 1, "approved"].mean()), 4),
        "approval_rate_not_thin": round(float(sc.loc[sc.thin_filer == 0, "approved"].mean()), 4),
        "default_rate_approved": round(float(sc.loc[sc.approved == 1, "default"].mean()), 4),
        "default_rate_declined": round(float(sc.loc[sc.approved == 0, "default"].mean()), 4),
        "monotonic_violations": violations,
    }
    t = test_metrics
    summary["charter"] = {
        "test_auc>=0.78": t["auc"] >= 0.78,
        "test_ks>=0.28": t["ks"] >= 0.28,
        "thin_lift>=0.03": t["thin_auc_lift"] >= 0.03,
        "psi<0.1": t["psi_trainval_vs_test"] < 0.1,
        "mono_loss<=0.01": None if mono_loss is None else mono_loss["loss"] <= 0.01,
    }

    use_experiment(cfg, "final")
    run_name = f"{name}__both__{res}__mono-{'none' if not mono else ('full' if not dropped else 'no-' + '-'.join(d.lower()[:12] for d in dropped))}{'__platt' if calibrate else ''}__final"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"step": "5", "stage": "final", "owner": "wonbin", "trained_on": "train+validation",
                         "evaluated_on": "test"})
        mlflow.log_params({"model": name, "feature_set": "both", "resampling": res, "calibrated": calibrate,
                           "dropped_constraints": ",".join(dropped) or "none", "seed": cfg["seed"],
                           "data_md5": data_md5(cfg),
                           **{f"hp_{k}": v for k, v in cfg["models"][name].items()}})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in t.items() if isinstance(v, (int, float))})
        if mono_loss:
            mlflow.log_metric(ev.KEY_MONO_LOSS, mono_loss["loss"])
        mlflow.log_metrics({"approval_rate": summary["approval_rate"],
                            "approval_rate_thin": summary["approval_rate_thin"]})
        mlflow.log_dict(summary, "final_summary.json")
        mlflow.log_text(grade_tbl.round(4).to_csv(), "grade_table.csv")
        mlflow.log_artifact(str(ROOT / "config" / "scoring.yaml"))
        mlflow.log_artifact(str(ROOT / "config" / "actionability.yaml"))
        summary["skops_trusted_types"] = log_final_model(final_est, X_tv.iloc[:5])
        summary["mlflow_run_name"] = run_name
        summary["mlflow_run_id"] = mlflow.active_run().info.run_id

    (ROOT / cfg["outputs"]["final_summary"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("winner", "test", "monotonic_loss_validation", "charter")}, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["compare", "monotonic", "final", "all"], default="all")
    stage = ap.parse_args().stage

    cfg = load_yaml("train.yaml")
    df = pd.read_csv(ROOT / cfg["data"])
    fs = feature_sets()
    fs = {"gmsc": fs["base"], "alt": fs["alt"], "both": fs["all"]}
    labels = make_split(df, cfg)
    save_split(labels, cfg)
    train, val, test = (df[labels == k] for k in ("train", "validation", "test"))
    folds = make_folds(train.reset_index(drop=True), cfg)
    setup_mlflow(cfg)

    out = cfg["outputs"]
    if stage in ("compare", "all"):
        stage_compare(train.reset_index(drop=True), folds, cfg, fs)
    comparison = pd.read_csv(ROOT / out["comparison"])
    if stage in ("monotonic", "all"):
        stage_monotonic(train, val, cfg, fs, comparison)
    if stage in ("final", "all"):
        monotonic = pd.read_csv(ROOT / out["monotonic"])
        stage_final(train, val, test, cfg, fs, comparison, monotonic)


if __name__ == "__main__":
    main()
