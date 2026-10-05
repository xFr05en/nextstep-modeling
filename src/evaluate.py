"""Evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def thin_filer_auc_lift(df: pd.DataFrame, target: str, base_features: list[str],
                        alt_features: list[str], thin_col: str = "thin_filer",
                        n_splits: int = 5, seed: int = 42) -> dict:
    """AUC gain from adding alternative features, overall and on thin-filer rows.

    Both models use the same stratified folds (stratified on target x thin_filer).
    LightGBM with default settings; it handles NaN natively, so no imputation here.
    """
    y = df[target].to_numpy()
    thin = df[thin_col].to_numpy() == 1
    strata = y * 2 + thin
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    feature_sets = {"base": base_features, "full": base_features + alt_features}

    rows = []
    for fold, (tr, te) in enumerate(skf.split(df, strata)):
        rec = {"fold": fold}
        for name, cols in feature_sets.items():
            model = LGBMClassifier(random_state=seed, verbose=-1)
            model.fit(df.iloc[tr][cols], y[tr])
            p = model.predict_proba(df.iloc[te][cols])[:, 1]
            rec[f"auc_{name}"] = roc_auc_score(y[te], p)
            t = thin[te]
            rec[f"thin_auc_{name}"] = roc_auc_score(y[te][t], p[t])
        rec["lift"] = rec["auc_full"] - rec["auc_base"]
        rec["thin_lift"] = rec["thin_auc_full"] - rec["thin_auc_base"]
        rows.append(rec)

    per_fold = pd.DataFrame(rows)
    summary = {
        k: {"mean": round(float(per_fold[k].mean()), 4), "std": round(float(per_fold[k].std()), 4)}
        for k in ["auc_base", "auc_full", "lift", "thin_auc_base", "thin_auc_full", "thin_lift"]
    }
    summary["per_fold_thin_lift"] = [round(float(v), 4) for v in per_fold["thin_lift"]]
    return summary


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])
