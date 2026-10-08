"""Shared test helpers. Data files are gitignored, so tests that need them skip by default.

Set REQUIRE_DATA=1 (Docker, CI) to make those tests fail instead of skip.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HINT = "run `python -m src.data`, `python -m src.simulator` and `python -m src.train --stage all` first"


def require(path: Path) -> Path:
    """Skip (default) or fail (REQUIRE_DATA=1) when a needed file is missing."""
    if not path.exists():
        msg = f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} not found: {HINT}"
        if os.environ.get("REQUIRE_DATA") == "1":
            pytest.fail(msg)
        pytest.skip(msg)
    return path


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data() -> pd.DataFrame:
    return _read_csv(require(ROOT / load_yaml("data.yaml")["processed_path"])).copy()


def sim_data() -> pd.DataFrame:
    return _read_csv(require(ROOT / load_yaml("simulator.yaml")["outputs"]["main"])).copy()


@lru_cache
def _load_model(path: Path):
    return joblib.load(path)


def final_model_path() -> Path:
    """Versioned final model file named in reports/final_summary.json (e.g. models/xgboost_v1.0.joblib)."""
    import json

    summary = require(ROOT / load_yaml("train.yaml")["outputs"]["final_summary"])
    return ROOT / json.loads(summary.read_text())["model_file"]


def final_model():
    return _load_model(require(final_model_path()))
