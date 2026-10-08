"""Evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
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


# ---- Step 5 metrics ----

# MLflow metric keys (shared with the serving code; see reports/mlflow_schema_en.md)
KEY_AUC = "auc"
KEY_KS = "ks"
KEY_THIN_AUC = "thin_auc"
KEY_THIN_GAIN = "thin_auc_gain"
KEY_PSI = "psi"
KEY_BRIER = "brier"
KEY_MONO_LOSS = "mono_auc_loss"
KEY_PRECISION = "precision"
KEY_RECALL = "recall"
KEY_F1 = "f1"


def predicted_default(pd_: np.ndarray, scoring_cfg: dict | None = None) -> np.ndarray:
    """1 = predicted default = rejected: score below the approval cutoff in config/scoring.yaml.

    The cutoff is a fixed rule from the config; no data (and never test data) is used to set it."""
    from src.scoring import load_config as load_scoring, pd_to_score, score_to_grade

    cfg = scoring_cfg or load_scoring()
    grades = score_to_grade(pd_to_score(pd_, cfg), cfg)
    return (~np.isin(grades, cfg["approval"]["approve_grades"])).astype(int)


def classification_metrics(y, pd_, scoring_cfg: dict | None = None) -> dict:
    """Precision, recall and F1 for the default class (positive = default) at the score cutoff."""
    pred = predicted_default(pd_, scoring_cfg)
    return {KEY_PRECISION: precision_score(y, pred, zero_division=0),
            KEY_RECALL: recall_score(y, pred, zero_division=0),
            KEY_F1: f1_score(y, pred, zero_division=0)}


def bootstrap_ci(stat, n: int, n_boot: int = 500, seed: int = 0) -> list[float]:
    """2.5 / 97.5 percentiles of stat(idx) over bootstrap resamples of row indices.

    For paired comparisons (e.g. thin-filer lift), stat computes both models on the same idx."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def ks_stat(y, score) -> float:
    """Kolmogorov-Smirnov: max gap between the score CDFs of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def psi(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index of `actual` scores vs `expected` (bins from expected quantiles)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected) + eps
    a = np.histogram(actual, edges)[0] / len(actual) + eps
    return float(np.sum((a - e) * np.log(a / e)))


def fold_metrics(y_te, p_te, p_tr, thin_te, scoring_cfg: dict | None = None) -> dict:
    """Metrics for one held-out fold. p_tr = scores on the training fold (for PSI)."""
    return {
        KEY_AUC: roc_auc_score(y_te, p_te),
        KEY_KS: ks_stat(y_te, p_te),
        KEY_THIN_AUC: roc_auc_score(y_te[thin_te], p_te[thin_te]),
        KEY_PSI: psi(p_tr, p_te),
        KEY_BRIER: brier_score_loss(y_te, p_te),
        **classification_metrics(y_te, p_te, scoring_cfg),
    }
