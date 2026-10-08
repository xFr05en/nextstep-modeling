"""German Credit secondary model (separate from the GMSC pipeline; shares no config or model).

Data: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5NC77 (CC BY 4.0). Settings in config/german.yaml.

Sex (from attribute 9) and foreign worker are never model features; they are kept for evaluation.
Encoding and scaling are pipeline steps, so they are fit on training folds only.

Run:  python -m src.german

Note: this model is not saved to models/. If it ever is, use the versioned name rule from
config/train.yaml, e.g. models/german_xgboost_v1.0.joblib.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import load_german
from src.data.preprocessor import make_onehot_encoder, make_scaler
from src.evaluate import ks_stat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "german.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict, path: Path | None = None, verbose: bool = False) -> pd.DataFrame:
    return load_german(path or ROOT / cfg["raw_path"], verbose=verbose)


def decode(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Readable labels, target as 1 = default, plus sex derived from attribute 9."""
    df = raw.copy()
    df["default"] = (df["target"] == cfg["bad_code"]).astype(int)
    df["sex"] = np.where(df["personal_status_sex"].isin(cfg["female_codes"]), "female", "male")
    for col, mapping in cfg["labels"].items():
        unknown = set(df[col]) - set(mapping)
        if unknown:
            raise ValueError(f"Unknown codes in {col}: {sorted(unknown)}")
        df[col] = df[col].map(mapping)
    return df.drop(columns="target")


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    excluded = set(cfg["exclude_from_features"]) | {"target"}
    numeric = [c for c in cfg["numeric"] if c not in excluded]
    categorical = [c for c in cfg["columns"] if c not in excluded and c not in cfg["numeric"]]
    return numeric, categorical


def onehot_name(feature: str, category) -> str:
    """One-hot column name without characters XGBoost rejects ([, ], <)."""
    cat = str(category).replace(">=", "ge ").replace("<", "lt ").replace("[", "(").replace("]", ")")
    return f"{feature}={cat}"


def build_pipeline(model: str, cfg: dict, constrained: bool = True) -> Pipeline:
    numeric, categorical = feature_lists(cfg)
    num_step = make_scaler() if model == "lr" else "passthrough"
    prep = ColumnTransformer(
        [("num", num_step, numeric),
         ("cat", make_onehot_encoder(feature_name_combiner=onehot_name), categorical)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    if model == "lr":
        est = LogisticRegression(**cfg["models"]["lr"])
    else:
        mono = cfg["monotone"] if constrained else {}
        est = XGBClassifier(**cfg["models"]["xgb"], random_state=cfg["cv"]["seed"], n_jobs=-1,
                            eval_metric="logloss", monotone_constraints=mono or None)
    return Pipeline([("prep", prep), ("model", est)])


def cross_validate(df: pd.DataFrame, cfg: dict) -> tuple[dict, pd.DataFrame]:
    numeric, categorical = feature_lists(cfg)
    X, y = df[numeric + categorical], df["default"].to_numpy()
    cv = cfg["cv"]
    rskf = RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=cv["seed"])
    variants = {"lr": ("lr", True), "xgb": ("xgb", True), "xgb_unconstrained": ("xgb", False)}
    oof = {k: np.zeros((cv["n_repeats"], len(df))) for k in variants}
    rows = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        rep = i // cv["n_splits"]
        rec = {"repeat": rep, "fold": i % cv["n_splits"]}
        for name, (model, constrained) in variants.items():
            pipe = build_pipeline(model, cfg, constrained).fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            oof[name][rep, te] = p
            rec[f"{name}_auc"] = roc_auc_score(y[te], p)
            rec[f"{name}_ks"] = ks_stat(y[te], p)
            rec[f"{name}_brier"] = brier_score_loss(y[te], p)
        rows.append(rec)
    folds = pd.DataFrame(rows)
    per_rep = folds.groupby("repeat").mean()

    def spread(col):
        return {"mean": round(float(per_rep[col].mean()), 4), "sd_across_repeats": round(float(per_rep[col].std()), 4),
                "fold_min": round(float(folds[col].min()), 4), "fold_max": round(float(folds[col].max()), 4)}

    metrics = {name: {m: spread(f"{name}_{m}") for m in ("auc", "ks", "brier")} for name in variants}
    diff = folds["xgb_auc"] - folds["lr_auc"]
    metrics["xgb_minus_lr_auc"] = {"mean": round(float(diff.mean()), 4), "sd_across_folds": round(float(diff.std()), 4),
                                   "share_of_folds_xgb_better": round(float((diff > 0).mean()), 4)}
    loss = folds["xgb_unconstrained_auc"] - folds["xgb_auc"]
    metrics["monotone_auc_loss"] = {"mean": round(float(loss.mean()), 4), "sd_across_folds": round(float(loss.std()), 4)}
    # Out-of-fold PD averaged over repeats: every value comes from models that never saw the row
    pd_avg = {k: v.mean(axis=0) for k, v in oof.items()}
    metrics["oof_avg_auc"] = {k: round(roc_auc_score(y, v), 4) for k, v in pd_avg.items()}
    return metrics, pd.DataFrame(pd_avg)


def describe_groups(df: pd.DataFrame) -> dict:
    age_band = pd.cut(df["age"], [0, 29, 49, 200], labels=["under_30", "30_to_49", "50_plus"])
    out = {}
    for name, g in {"sex": df["sex"], "age_band": age_band, "foreign_worker": df["foreign_worker"],
                    "personal_status": df["personal_status_sex"]}.items():
        t = df.groupby(g, observed=True)["default"].agg(["size", "mean"])
        out[name] = {str(k): {"rows": int(r["size"]), "default_rate": round(float(r["mean"]), 4)} for k, r in t.iterrows()}
    return out


def main() -> None:
    cfg = load_config()
    df = decode(load_raw(cfg), cfg)
    metrics, pd_avg = cross_validate(df, cfg)
    c = cfg["cost_matrix"]
    metrics.update({
        "rows": len(df), "default_rate": round(float(df["default"].mean()), 4),
        "features": dict(zip(["numeric", "categorical"], feature_lists(cfg))),
        "monotone": cfg["monotone"], "cv": cfg["cv"], "groups": describe_groups(df),
        "cost_matrix": c,
        # With calibrated PD, approving costs approve_bad * PD and rejecting costs reject_good * (1 - PD)
        "cost_break_even_pd": round(c["reject_good"] / (c["approve_bad"] + c["reject_good"]), 4),
        "source": cfg["source"],
    })
    oof = pd.DataFrame({
        "row_id": np.arange(len(df)), "pd_xgb": pd_avg["xgb"].round(6), "pd_lr": pd_avg["lr"].round(6),
        "default": df["default"], "sex": df["sex"], "age": df["age"],
        "personal_status": df["personal_status_sex"], "foreign_worker": df["foreign_worker"],
    })
    out = ROOT / cfg["outputs"]["oof"]
    out.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out, index=False)
    (ROOT / cfg["outputs"]["metrics"]).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("lr", "xgb", "xgb_unconstrained", "xgb_minus_lr_auc",
                                              "monotone_auc_loss", "oof_avg_auc")}, indent=2))


if __name__ == "__main__":
    main()
