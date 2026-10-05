"""Alternative-data simulator (Step 3): Gaussian copula with a default part and a credit part.

For each alternative variable j:
    z_j = -(a_j * T + b * w_j * C) + n_j * e_j
    T   latent default score: liability draw consistent with the 0/1 target
    C   credit-behavior score from past-due counts and revolving utilization
    e_j independent noise
z_j is turned into a uniform by rank, then into the variable's marginal (copula step).
a_j is calibrated so the LATENT correlation r(z_j, T) equals target_corr (charter rule 0.3 to 0.5).
Observed Pearson and Spearman r with the 0/1 target are reported alongside.

Run:  python -m src.simulator   (needs data/processed/gmsc_clean.csv from src.data)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score

from src.evaluate import pearson
from src.features import add_thin_filer_flag

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "simulator.yaml"
ONTIME_PAIR = ("telecom_ontime_rate", "utility_ontime_rate")


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- Building blocks ----

def latent_default(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N(0,1) liability: defaulters above the (1 - default rate) quantile, others below."""
    t = stats.norm.ppf(1 - y.mean())
    out = np.empty(len(y))
    d = y == 1
    out[d] = stats.truncnorm.rvs(t, np.inf, size=d.sum(), random_state=rng)
    out[~d] = stats.truncnorm.rvs(-np.inf, t, size=(~d).sum(), random_state=rng)
    return out


def normal_scores(x: pd.Series) -> np.ndarray:
    """Rank-based normal scores (ties share a rank). NaN stays NaN."""
    r = x.rank(method="average")
    return stats.norm.ppf(((r - 0.5) / x.notna().sum()).to_numpy())


def credit_score(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Standardized credit-behavior score; higher = worse behavior."""
    cl = cfg["credit_link"]
    pastdue = df[cl["pastdue_columns"]].sum(axis=1, min_count=1)
    # Special-code rows (96/98) default at ~55%: rank them as the worst past-due behavior
    pastdue = pastdue.where(df["pastdue_special_code"] == 0, np.inf)
    z_pd = normal_scores(pastdue)
    # Utilization outliers default at about the average rate: neutral score
    z_util = np.nan_to_num(normal_scores(df[cl["util_column"]]), nan=0.0)
    c = z_pd + z_util
    return (c - c.mean()) / c.std()


def to_marginal(u: np.ndarray, spec: dict) -> np.ndarray:
    """Map uniforms in (0, 1) to the variable's marginal distribution."""
    p = spec.get("params", {})
    kind = spec["marginal"]
    if kind == "beta":
        x = stats.beta.ppf(u, p["a"], p["b"])
    elif kind == "gamma":
        x = stats.gamma.ppf(u, p["shape"], scale=p["scale"])
    elif kind == "binomial":
        x = stats.binom.ppf(u, p["n"], p["p"])
    elif kind == "discrete":
        vals = np.asarray(spec["values"], float)
        x = vals[np.minimum((u * len(vals)).astype(int), len(vals) - 1)]
    else:
        raise ValueError(f"Unknown marginal: {kind}")
    if "round_to" in spec:
        k = round(1 / spec["round_to"])  # 0.0833 -> 12 steps, so 1.0 stays exactly 1.0
        x = np.round(x * k) / k
    if "clip" in spec:
        x = np.clip(x, *spec["clip"])
    if spec.get("integer"):
        x = np.round(x)
    return x


def latent_score(a: float, b: float, noise: float, T, C, e) -> np.ndarray:
    return -(a * T + b * C) + noise * e


def to_variable(z: np.ndarray, spec: dict, age, cfg) -> np.ndarray:
    """Copula step: rank -> uniform -> marginal, then the tenure age cap."""
    u = (stats.rankdata(z) - 0.5) / len(z)
    x = to_marginal(u, spec)
    if spec.get("age_cap"):
        x = np.minimum(x, np.maximum(age - cfg["tenure_min_age"], 0) * 12)
    return x


def calibrate(spec: dict, b: float, noise: float, T, C, e, cal: dict) -> tuple[float, np.ndarray]:
    """Bisection on a so that the latent r(z, T) hits spec['target_corr'] (negative).

    z includes the credit part, so the total latent correlation is calibrated.
    """
    want = abs(spec["target_corr"])

    def gap(a):
        z = latent_score(a, b, noise, T, C, e)
        return -pearson(z, T) - want, z

    g, z = gap(0.0)
    if g > 0:
        raise ValueError(f"Credit link alone gives latent |r| above target ({g + want:.3f}); lower b")
    lo, hi = 0.0, cal["a_max"]
    for _ in range(cal["max_iter"]):
        a = (lo + hi) / 2
        g, z = gap(a)
        if abs(g) <= cal["tolerance"]:
            return a, z
        lo, hi = (a, hi) if g < 0 else (lo, a)
    raise ValueError(f"Calibration did not converge (gap {g:.4f})")


# ---- Main entry ----

def simulate(df: pd.DataFrame, cfg: dict, b: float | None = None,
             target: str = "SeriousDlqin2yrs") -> tuple[pd.DataFrame, dict]:
    """Add the alternative variables and gender to df. Returns (new frame, info)."""
    b = cfg["credit_link"]["b"] if b is None else b
    specs = cfg["variables"]
    names = list(specs)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(cfg["seed"]).spawn(len(names) + 2)]

    y = df[target].to_numpy()
    age = df["age"].to_numpy()
    T = latent_default(y, rngs[0])
    C = credit_score(df, cfg)
    E = {n: rngs[i + 2].standard_normal(len(df)) for i, n in enumerate(names)}
    noise = {n: float(specs[n].get("noise", 1.0)) for n in names}

    for _round in range(cfg["noise_max_rounds"] + 1):
        params, X, Z = {}, {}, {}
        for n in names:
            bj = b * specs[n].get("credit_weight", 1.0)
            a, Z[n] = calibrate(specs[n], bj, noise[n], T, C, E[n], cfg["calibration"])
            X[n] = to_variable(Z[n], specs[n], age, cfg)
            params[n] = {"a": round(a, 4), "b": round(bj, 4), "noise": round(noise[n], 2)}
        inter = pd.DataFrame(X).corr()
        if inter.loc[ONTIME_PAIR].item() <= cfg["max_inter_corr"]:
            break
        for n in ONTIME_PAIR:
            noise[n] += cfg["noise_step"]
    else:
        raise ValueError("On-time rates still above max_inter_corr after max noise rounds")

    off_diag = inter.where(~np.eye(len(names), dtype=bool))
    if (off_diag.abs() > cfg["max_inter_corr"]).any().any():
        raise ValueError(f"Inter-correlation above cap:\n{inter.round(3)}")

    out = df.copy()
    for n in names:
        out[n] = X[n]
    out["gender_female"] = (rngs[1].random(len(df)) < cfg["gender"]["p_female"]).astype(int)

    cl = cfg["credit_link"]
    pastdue_sum = df[cl["pastdue_columns"]].sum(axis=1)
    info = {
        "b": b,
        "noise_rounds": _round,
        "params": params,
        "corr_target_latent": {n: round(pearson(Z[n], T), 4) for n in names},
        "corr_target": {n: round(pearson(X[n], y), 4) for n in names},
        "corr_target_spearman": {n: round(float(stats.spearmanr(X[n], y)[0]), 4) for n in names},
        "auc_single": {n: round(float(roc_auc_score(y, -X[n])), 4) for n in names},
        "corr_credit_score": {n: round(pearson(X[n], C), 4) for n in names},
        "corr_pastdue_sum": {n: round(float(pd.Series(X[n]).corr(pastdue_sum)), 4) for n in names},
        "corr_utilization": {n: round(float(pd.Series(X[n]).corr(df[cl["util_column"]])), 4) for n in names},
        "corr_age": {n: round(pearson(X[n], age), 4) for n in names},
        "inter_corr": inter.round(4).to_dict(),
        "gender_female_share": round(float(out["gender_female"].mean()), 4),
        "gender_corr_target": round(pearson(out["gender_female"], y), 4),
    }
    return out, info


def main() -> None:
    cfg = load_config()
    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml", encoding="utf-8"))
    df = pd.read_csv(ROOT / data_cfg["processed_path"])
    df = add_thin_filer_flag(df, cfg["thin_filer"]["rule"])

    summary = {"thin_filer_share": round(float(df["thin_filer"].mean()), 4),
               "thin_filer_rows": int(df["thin_filer"].sum())}
    for key, b in [("main", cfg["credit_link"]["b"]), ("target_only", 0.0)]:
        sim, info = simulate(df, cfg, b=b, target=data_cfg["target"])
        path = ROOT / cfg["outputs"][key]
        sim.to_csv(path, index=False)
        summary[key] = info
        print(f"{key}: b={b}, saved {len(sim)} rows to {path.relative_to(ROOT)}")

    out = ROOT / cfg["outputs"]["summary"]
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
