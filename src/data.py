"""Load and clean the GMSC dataset (Step 1).

Rules are read from config/data.yaml. This module only fixes clearly wrong
values and adds flag columns. Missing values are left as NaN on purpose:
imputation is fit on training folds later (src/features.py) to avoid leakage.

Run:  python -m src.data
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "data.yaml"

FLAG_COLUMNS = ["pastdue_special_code", "income_missing", "income_zero", "util_outlier"]


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / cfg["raw_path"])
    if cfg["id_column"] in df.columns:
        df = df.drop(columns=cfg["id_column"])
    return df


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
