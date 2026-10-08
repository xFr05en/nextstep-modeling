"""Preprocessing: GMSC cleaning rules, in-fold building blocks, and the train/validation/test split.

Cleaning (rules in config/data.yaml) only fixes clearly wrong values and adds flag columns.
Missing values are left as NaN on purpose: the imputer, scaler, encoder and LR transform below
are pipeline steps, so they are fit on training folds only (no leakage).

Run:  python -m src.data   (or python -m src.data.preprocessor)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.loader import load_gmsc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "data.yaml"

FLAG_COLUMNS = ["pastdue_special_code", "income_missing", "income_zero", "util_outlier"]


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Raw GMSC via the loader (prints column names, dtypes and missing ratios)."""
    return load_gmsc(ROOT / cfg["raw_path"] if cfg else None, verbose=verbose)


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply cleaning rules. Returns the cleaned frame and a summary of what changed."""
    df = df.copy()
    target = cfg["target"]
    log: dict = {"rows_raw": int(len(df))}

    # Count duplicates on the raw values; masking below would create extra matches
    raw_dup = df.duplicated()
    log["duplicate_rows_raw"] = int(raw_dup.sum())
    if cfg.get("drop_duplicates", False):
        df = df.loc[~raw_dup]

    # 1. Past-due special codes (96/98) -> NaN + flag
    pd_cols = cfg["pastdue_columns"]
    special = df[pd_cols].isin(cfg["pastdue_special_codes"]).any(axis=1)
    df["pastdue_special_code"] = special.astype(int)
    df[pd_cols] = df[pd_cols].mask(df[pd_cols].isin(cfg["pastdue_special_codes"]))
    log["pastdue_special_code_rows"] = int(special.sum())

    # 2. Income flags. Missing stays NaN; zero is kept as a real value.
    income = df["MonthlyIncome"]
    df["income_missing"] = income.isna().astype(int)
    df["income_zero"] = (income == 0).astype(int)
    log["income_missing_rows"] = int(df["income_missing"].sum())
    log["income_zero_rows"] = int(df["income_zero"].sum())

    # 3. DebtRatio is a raw amount (not a ratio) when income is missing or 0
    bad_ratio = income.isna() | (income == 0)
    df.loc[bad_ratio, "DebtRatio"] = np.nan
    log["debtratio_set_missing_rows"] = int(bad_ratio.sum())

    # 4. Revolving utilization above threshold -> NaN + flag
    util_col, thr = cfg["util_column"], cfg["util_outlier_threshold"]
    outlier = df[util_col] > thr
    df["util_outlier"] = outlier.astype(int)
    df.loc[outlier, util_col] = np.nan
    log["util_outlier_rows"] = int(outlier.sum())
    log["util_between_1_and_threshold_rows"] = int(((df[util_col] > 1) & (df[util_col] <= thr)).sum())

    # 5. Drop impossible ages
    bad_age = df["age"] < cfg["min_age"]
    log["age_dropped_rows"] = int(bad_age.sum())
    df = df.loc[~bad_age]

    # Rows that became identical only because values were set to NaN (informational)
    log["duplicate_rows_after_cleaning"] = int(df.duplicated().sum())

    df = df.reset_index(drop=True)
    log["rows_clean"] = int(len(df))
    log["default_rate"] = round(float(df[target].mean()), 4)
    log["missing_pct_after"] = {
        c: round(float(v) * 100, 2) for c, v in df.isna().mean().items() if v > 0
    }
    log["default_rate_by_flag"] = {
        f: {str(k): round(float(v), 4) for k, v in df.groupby(f)[target].mean().items()}
        for f in FLAG_COLUMNS
    }
    return df, log


# ---- In-fold building blocks (pipeline steps; fit on training folds only) ----

def make_imputer() -> SimpleImputer:
    """Missing values: median of the training fold."""
    return SimpleImputer(strategy="median")


def make_scaler() -> StandardScaler:
    """Numeric scaling (LR, and before SMOTE because it uses distances)."""
    return StandardScaler()


def make_onehot_encoder(feature_name_combiner=None) -> OneHotEncoder:
    """Categorical encoding. GMSC has no categorical columns; the German Credit model uses this."""
    kw = {"feature_name_combiner": feature_name_combiner} if feature_name_combiner else {}
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kw)


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


# ---- Split ----

def make_split(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Stratified (stratify=y) train / validation / test labels, e.g. 70:15:15. Index = df.index.
    cfg: config/train.yaml (split ratios, seed, target)."""
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
    """data/processed/split.csv (row_id, split): defines train / validation / test for the whole team."""
    out = ROOT / cfg["outputs"]["split"]
    pd.DataFrame({"row_id": labels.index, "split": labels.to_numpy()}).to_csv(out, index=False)


def main() -> None:
    cfg = load_config()
    df, log = clean(load_raw(cfg), cfg)
    out = ROOT / cfg["processed_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summary = ROOT / cfg["summary_path"]
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(log, indent=2))
    print(f"Saved {len(df)} rows to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
