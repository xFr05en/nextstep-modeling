"""Feature helpers shared by the simulator and the training code."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def is_thin_filer_proxy(df: pd.DataFrame, rule: str) -> pd.Series:
    """Our thin-filer proxy (used for every official number), a pandas query rule from
    config/simulator.yaml: NumberOfOpenCreditLinesAndLoans <= 2 and NumberRealEstateLoansOrLines == 0."""
    return df.eval(rule).astype(int)


def is_thin_filer_mission(df: pd.DataFrame, pastdue_columns: list[str], card_history_column: str | None = None,
                          min_missing: int = 2, min_card_months: int = 12) -> pd.Series:
    """The mission's thin-filer rule: at least `min_missing` of the past-due columns are missing
    OR card history is shorter than `min_card_months` months.

    GMSC has no card-history column, so with card_history_column=None only the missing-columns
    part applies. On the real GMSC file this flags 0% of raw rows, and after cleaning only the
    269 special-code (96/98) rows, which are the highest-risk group, not thin files."""
    flag = df[pastdue_columns].isna().sum(axis=1) >= min_missing
    if card_history_column is not None and card_history_column in df.columns:
        flag |= df[card_history_column] < min_card_months
    return flag.astype(int)


def add_thin_filer_flag(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Add `thin_filer` (0/1) with the proxy rule, e.g. from simulator.yaml."""
    df = df.copy()
    df["thin_filer"] = is_thin_filer_proxy(df, rule)
    return df


def feature_sets(actionability_path: Path | str = ROOT / "config" / "actionability.yaml",
                 simulator_path: Path | str = ROOT / "config" / "simulator.yaml") -> dict:
    """Model feature lists. Gender is protected and simulated, so it is never a model feature.
    Features with `model_feature: false` in the YAML stay in the data but are not model inputs."""
    with open(actionability_path, encoding="utf-8") as f:
        feats = yaml.safe_load(f)["features"]
    with open(simulator_path, encoding="utf-8") as f:
        alt = list(yaml.safe_load(f)["variables"])
    use = {k for k, v in feats.items() if v.get("model_feature", True)}
    base = [k for k, v in feats.items() if v["source"] in ("gmsc", "derived") and k in use]
    alt = [k for k in alt if k in use]
    return {"base": base, "alt": alt, "all": base + alt}
