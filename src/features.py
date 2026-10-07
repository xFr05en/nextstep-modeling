"""Feature helpers shared by the simulator and the training code."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def add_thin_filer_flag(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Add `thin_filer` (0/1) using a pandas query rule, e.g. from simulator.yaml."""
    df = df.copy()
    df["thin_filer"] = df.eval(rule).astype(int)
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
