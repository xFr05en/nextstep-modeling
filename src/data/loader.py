"""Load the raw datasets and print their schema (column names, dtypes, missing ratios).

Paths come from config/data.yaml (GMSC) and config/german.yaml (German Credit).
The schema is printed to stdout by default, so it is visible in the console.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def describe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: name, dtype, missing ratio."""
    return pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                         "missing_ratio": df.isna().mean().round(4).to_numpy()})


def print_schema(df: pd.DataFrame, name: str) -> None:
    print(f"[loader] {name}: {len(df):,} rows x {df.shape[1]} columns")
    print(describe_schema(df).to_string(index=False))


def load_gmsc(path: Path | str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Give Me Some Credit (cs-training.csv), with the unnamed row-id column dropped."""
    cfg = _load_yaml("data.yaml")
    df = pd.read_csv(Path(path) if path else ROOT / cfg["raw_path"])
    if cfg["id_column"] in df.columns:
        df = df.drop(columns=cfg["id_column"])
    if verbose:
        print_schema(df, "Give Me Some Credit")
    return df


def load_german(path: Path | str | None = None, verbose: bool = True) -> pd.DataFrame:
    """UCI Statlog German Credit (german.data), raw codes with column names from config/german.yaml."""
    cfg = _load_yaml("german.yaml")
    df = pd.read_csv(Path(path) if path else ROOT / cfg["raw_path"], sep=r"\s+", header=None,
                     names=cfg["columns"])
    if verbose:
        print_schema(df, "German Credit")
    return df
