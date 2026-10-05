"""Map calibrated PD to a 0-1000 credit score, a grade A to E, and an approval decision.

All numbers come from config/scoring.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "scoring.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def pd_to_score(pd_: np.ndarray, cfg: dict) -> np.ndarray:
    s = cfg["score"]
    p = np.clip(np.asarray(pd_, float), *s["pd_clip"])
    odds = (1 - p) / p
    score = s["base_score"] + s["pdo"] / np.log(2) * np.log(odds / s["base_odds"])
    return np.clip(np.round(score), s["min"], s["max"]).astype(int)


def score_to_grade(score: np.ndarray, cfg: dict) -> np.ndarray:
    grades = sorted(cfg["grades"], key=lambda g: -g["min_score"])
    out = np.full(len(score), grades[-1]["grade"], dtype=object)
    for g in reversed(grades):  # assign from worst to best so better grades overwrite
        out[np.asarray(score) >= g["min_score"]] = g["grade"]
    return out


def approve(grade: np.ndarray, cfg: dict) -> np.ndarray:
    return np.isin(grade, cfg["approval"]["approve_grades"]).astype(int)


def score_frame(pd_: np.ndarray, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    score = pd_to_score(pd_, cfg)
    grade = score_to_grade(score, cfg)
    return pd.DataFrame({"pd": pd_, "score": score, "grade": grade, "approved": approve(grade, cfg)})
