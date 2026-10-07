"""Alternative-data simulator: Gaussian copula with a default part, a credit part and a shared factor.

For each alternative variable j:
    z_j = -(a_j * T + b * w_j * C) + g * F + n_j * e_j
    T   latent default score: liability draw consistent with the 0/1 target
    C   credit-behavior score from past-due counts and revolving utilization (no age, no gender)
    F   one random factor shared by the mission variables only (g = 0 for the others)
    e_j independent noise
z_j is turned into a uniform by rank, then into the variable's marginal (copula step).
a_j is calibrated per variable (calibrate_on in config):
    observed: Pearson r(final x_j, 0/1 target) = target_corr   (the 5 mission variables)
    latent:   r(z_j, T) = target_corr                           (the 3 extra variables)
g is searched so the inter-correlation of the mission variables hits shared_factor.target.

Note: the target is used for every row, including rows later held out for testing,
as in the mission's own example simulator. GMSC-only results are the realistic reference.

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
    elif kind == "poisson":
        x = stats.poisson.ppf(u, p["mu"])
    elif kind == "discrete":
        vals = np.asarray(spec["values"], float)
        x = vals[np.minimum((u * len(vals)).astype(int), len(vals) - 1)]
    else:
        raise ValueError(f"Unknown marginal: {kind}")
    if "scale" in spec:
        x = x * spec["scale"]
    if "decimals" in spec:
        x = np.round(x, spec["decimals"])
    if "round_to" in spec:
        k = round(1 / spec["round_to"])  # 0.0833 -> 12 steps, so 1.0 stays exactly 1.0
        x = np.round(x * k) / k
    if "clip" in spec:
        x = np.clip(x, *spec["clip"])
    if spec.get("integer"):
        x = np.round(x)
    return x


def latent_score(a: float, b: float, noise: float, T, C, e, g: float = 0.0, F=None) -> np.ndarray:
    z = -(a * T + b * C) + noise * e
    return z if F is None or g == 0 else z + g * F


def to_variable(z: np.ndarray, spec: dict, age, cfg) -> np.ndarray:
    """Copula step: rank -> uniform -> marginal, then the tenure age cap."""
    u = (stats.rankdata(z) - 0.5) / len(z)
    x = to_marginal(u, spec)
    if spec.get("age_cap"):
        x = np.minimum(x, np.maximum(age - cfg["tenure_min_age"], 0) * 12)
    return x


def calibrate(spec: dict, b: float, noise: float, T, C, e, cal: dict, *, y=None, age=None, cfg=None,
              g: float = 0.0, F=None) -> tuple[float, np.ndarray, np.ndarray]:
    """Bisection on a so the variable hits spec['target_corr'] (negative), on the scale set by
    spec['calibrate_on']. The credit part and shared factor are part of z, so the total is calibrated.
    Returns (a, z, x)."""
    want = abs(spec["target_corr"])
    observed = spec.get("calibrate_on", "latent") == "observed"
    tol = cal["tolerance_observed"] if observed else cal["tolerance_latent"]

    def gap(a):
        z = latent_score(a, b, noise, T, C, e, g, F)
        x = to_variable(z, spec, age, cfg)
        r = pearson(x, y) if observed else pearson(z, T)
        return -r - want, z, x

    gp, z, x = gap(0.0)
    if gp > 0:
        raise ValueError(f"Credit link / shared factor alone give |r| above target ({gp + want:.3f})")
    lo, hi = 0.0, cal["a_max"]
    for _ in range(cal["max_iter"]):
        a = (lo + hi) / 2
        gp, z, x = gap(a)
        if abs(gp) <= tol:
            return a, z, x
        lo, hi = (a, hi) if gp < 0 else (lo, a)
    raise ValueError(f"Calibration did not converge (gap {gp:.4f})")


def _pair_stat(X: dict, names: list[str], stat: str) -> float:
    c = pd.DataFrame({n: X[n] for n in names}).corr().to_numpy()
    off = c[~np.eye(len(names), dtype=bool)]
    return float(off.max() if stat == "max" else off.mean())


# ---- Main entry ----

def simulate(df: pd.DataFrame, cfg: dict, b: float | None = None,
             target: str = "SeriousDlqin2yrs") -> tuple[pd.DataFrame, dict]:
    """Add the alternative variables and gender to df. Returns (new frame, info)."""
    b = cfg["credit_link"]["b"] if b is None else b
    specs = cfg["variables"]
    names = list(specs)
    shared = [n for n in names if specs[n].get("shared_factor")]
    # Child seeds by position: adding variables at the end keeps earlier draws unchanged.
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(cfg["seed"]).spawn(len(names) + 3)]

    y = df[target].to_numpy()
    age = df["age"].to_numpy()
    T = latent_default(y, rngs[0])
    C = credit_score(df, cfg)
    E = {n: rngs[i + 2].standard_normal(len(df)) for i, n in enumerate(names)}
    F = rngs[len(names) + 2].standard_normal(len(df))
    cal = cfg["calibration"]
    kw = {"y": y, "age": age, "cfg": cfg}

    def run(names_, g):
        out = {}
        for n in names_:
            bj = b * specs[n].get("credit_weight", 1.0)
            gn = g if n in shared else 0.0
            a, z, x = calibrate(specs[n], bj, float(specs[n].get("noise", 1.0)), T, C, E[n], cal, g=gn, F=F, **kw)
            out[n] = (a, bj, gn, z, x)
        return out

    # Search the shared-factor weight g (inter-correlation rises with g)
    sf = cfg.get("shared_factor")
    g, sf_info = 0.0, None
    if shared and sf:
        stat, want = sf["target"]["stat"], sf["target"]["value"]
        res = run(shared, 0.0)
        at_zero = _pair_stat({n: v[4] for n, v in res.items()}, shared, stat)
        feasible = at_zero <= want + sf["tolerance"]
        if feasible and at_zero < want - sf["tolerance"]:
            lo, hi = 0.0, sf["g_max"]
            for _ in range(sf["max_iter"]):
                g = (lo + hi) / 2
                res = run(shared, g)
                got = _pair_stat({n: v[4] for n, v in res.items()}, shared, stat)
                if abs(got - want) <= sf["tolerance"]:
                    break
                lo, hi = (g, hi) if got < want else (lo, g)
        Xs = {n: v[4] for n, v in res.items()}
        sf_info = {"target": sf["target"], "g": round(g, 4), "feasible": bool(feasible),
                   "value_at_g0": round(at_zero, 4),
                   "achieved_mean": round(_pair_stat(Xs, shared, "mean"), 4),
                   "achieved_max": round(_pair_stat(Xs, shared, "max"), 4)}
        results = {**res, **run([n for n in names if n not in shared], 0.0)}
    else:
        results = run(names, 0.0)

    X = {n: results[n][4] for n in names}
    Z = {n: results[n][3] for n in names}
    params = {n: {"a": round(results[n][0], 4), "b": round(results[n][1], 4), "g": round(results[n][2], 4),
                  "noise": float(specs[n].get("noise", 1.0)), "calibrate_on": specs[n].get("calibrate_on", "latent")}
              for n in names}
    inter = pd.DataFrame(X)[names].corr()
    off_diag = inter.where(~np.eye(len(names), dtype=bool))
    if (off_diag.abs() > cfg["max_inter_corr"]).any().any():
        raise ValueError(f"Inter-correlation above cap:\n{inter.round(3)}")

    out = df.copy()
    for n in names:
        out[n] = X[n]
    out["gender_female"] = (rngs[1].random(len(df)) < cfg["gender"]["p_female"]).astype(int)

    cl = cfg["credit_link"]
    pastdue_sum = df[cl["pastdue_columns"]].sum(axis=1)
    bands = cfg["age_bands"]
    band = pd.cut(df["age"], bands["edges"], labels=bands["labels"])
    info = {
        "b": b,
        "shared_factor": sf_info,
        "params": params,
        "corr_target_latent": {n: round(pearson(Z[n], T), 4) for n in names},
        "corr_target": {n: round(pearson(X[n], y), 4) for n in names},
        "corr_target_spearman": {n: round(float(stats.spearmanr(X[n], y)[0]), 4) for n in names},
        "auc_single": {n: round(float(roc_auc_score(y, -X[n])), 4) for n in names},
        "corr_credit_score": {n: round(pearson(X[n], C), 4) for n in names},
        "corr_pastdue_sum": {n: round(float(pd.Series(X[n]).corr(pastdue_sum)), 4) for n in names},
        "corr_utilization": {n: round(float(pd.Series(X[n]).corr(df[cl["util_column"]])), 4) for n in names},
        "corr_age": {n: round(pearson(X[n], age), 4) for n in names},
        "mean_by_age_band": {n: {str(k): round(float(v), 3) for k, v in pd.Series(X[n]).groupby(band.to_numpy(), observed=True).mean().items()}
                             for n in names},
        "inter_corr": inter.round(4).to_dict(),
        "max_pair_corr": round(float(off_diag.abs().max().max()), 4),
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
