"""Exploratory data analysis of the cleaned GMSC data (Step 2).

Reads data/processed/gmsc_clean.csv (run `python -m src.data` first) and writes:
  reports/figures/<name>_ko.png and <name>_en.png  (same figure, two languages)
  reports/eda_summary.json                         (numbers used in the EDA report)

Run from the project root:  python notebooks/eda.py
Nothing in src/ depends on this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"

# ---- Style (reference palette, light surface) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # series 1: non-default / flag = 0 / main series
ORANGE = "#eb6834"   # series 2: default / flag = 1
GRAY_BAR = "#b9b8b2" # missing-value bin
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#184f95", "#6da7ec", "#f0efec", "#ec7b7a", "#9c2222"]
)
FONTS = {"ko": "AppleGothic", "en": "DejaVu Sans"}

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines", "age",
    "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate", "DebtRatio", "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines", "NumberOfDependents",
]
FLAGS = ["pastdue_special_code", "income_missing", "income_zero", "util_outlier"]
# Count features get integer bins capped at this value ("3+"); others get deciles.
COUNT_CAPS = {
    "NumberOfTime30-59DaysPastDueNotWorse": 3, "NumberOfTime60-89DaysPastDueNotWorse": 3,
    "NumberOfTimes90DaysLate": 3, "NumberRealEstateLoansOrLines": 4, "NumberOfDependents": 4,
}
SKEWED = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio", "MonthlyIncome"]
LOG_TICKS = {"DebtRatio": (0, 0.25, 0.5, 1, 2, 4), "MonthlyIncome": (0, 10, 100, 1_000, 10_000, 100_000)}

# Short plot labels (full labels live in config/actionability.yaml)
SHORT = {
    "en": {
        "RevolvingUtilizationOfUnsecuredLines": "Revolving utilization",
        "age": "Age",
        "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late (count)",
        "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late (count)",
        "NumberOfTimes90DaysLate": "90+ days late (count)",
        "DebtRatio": "Debt ratio",
        "MonthlyIncome": "Monthly income",
        "NumberOfOpenCreditLinesAndLoans": "Open loans and credit lines",
        "NumberRealEstateLoansOrLines": "Real estate loans",
        "NumberOfDependents": "Dependents",
        "pastdue_special_code": "Past-due code 96/98",
        "income_missing": "Income missing",
        "income_zero": "Income = 0",
        "util_outlier": "Utilization > 10",
    },
    "ko": {
        "RevolvingUtilizationOfUnsecuredLines": "리볼빙 사용률",
        "age": "연령",
        "NumberOfTime30-59DaysPastDueNotWorse": "30~59일 연체 횟수",
        "NumberOfTime60-89DaysPastDueNotWorse": "60~89일 연체 횟수",
        "NumberOfTimes90DaysLate": "90일 이상 연체 횟수",
        "DebtRatio": "부채비율",
        "MonthlyIncome": "월 소득",
        "NumberOfOpenCreditLinesAndLoans": "보유 대출·신용한도 수",
        "NumberRealEstateLoansOrLines": "부동산 담보대출 수",
        "NumberOfDependents": "부양가족 수",
        "pastdue_special_code": "연체 특수코드 96/98",
        "income_missing": "소득 결측",
        "income_zero": "소득 0",
        "util_outlier": "사용률 10 초과",
    },
}
TEXT = {
    "en": {
        "non_default": "No default", "default": "Default", "missing": "Missing",
        "share_rows": "Share of rows (%)", "rows": "rows",
        "class_title": "Class balance: 6.7% of borrowers default",
        "default_rate": "Default rate (%)", "overall": "Overall",
        "bin_title": "Default rate by feature range",
        "bin_x": "bin (upper bound)",
        "dist_title": "Distribution of skewed features, default vs. no default",
        "density": "Share of group (%)", "log_x": "log scale, up to 99th percentile",
        "linear_x": "linear scale, values above 1.5 grouped at the right edge",
        "corr_title": "Rank correlation between features (Spearman)",
        "missing_title": "Missing values after cleaning",
        "missing_pct": "Missing (%)",
        "flag_title": "Default rate by flag",
        "flag0": "Flag = 0", "flag1": "Flag = 1",
        "target_title": "Correlation with default (Pearson r)",
        "target_band": "Alternative-data target |r| 0.3 to 0.5",
        "abs_r": "|r| with default",
        "thin_title_size": "Group size", "thin_title_rate": "Default rate",
        "thin": "Thin-filer", "not_thin": "Others",
    },
    "ko": {
        "non_default": "정상", "default": "부도", "missing": "결측",
        "share_rows": "전체 대비 비율 (%)", "rows": "행",
        "class_title": "클래스 분포: 대출자의 6.7%가 부도",
        "default_rate": "부도율 (%)", "overall": "전체 평균",
        "bin_title": "구간별 부도율",
        "bin_x": "구간 (상한값)",
        "dist_title": "치우친 변수의 분포: 부도 대 정상",
        "density": "집단 내 비율 (%)", "log_x": "로그 척도, 99 백분위수까지",
        "linear_x": "선형 척도, 1.5 초과 값은 오른쪽 끝에 합산",
        "corr_title": "변수 간 순위 상관 (Spearman)",
        "missing_title": "정제 후 결측 비율",
        "missing_pct": "결측 (%)",
        "flag_title": "플래그별 부도율",
        "flag0": "플래그 = 0", "flag1": "플래그 = 1",
        "target_title": "부도와의 상관계수 (Pearson r)",
        "target_band": "대체 데이터 목표 |r| 0.3~0.5",
        "abs_r": "부도와의 |r|",
        "thin_title_size": "집단 크기", "thin_title_rate": "부도율",
        "thin": "씬파일러", "not_thin": "그 외",
    },
}


def set_style(lang: str) -> None:
    plt.rcParams.update({
        "font.family": FONTS[lang],
        "axes.unicode_minus": False,  # AppleGothic has no Unicode minus glyph
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "figure.dpi": 150,
    })


def save(fig, name: str, lang: str) -> None:
    fig.savefig(FIG_DIR / f"{name}_{lang}.png", bbox_inches="tight")
    plt.close(fig)


def fmt_edge(x: float, decimals: int = 2) -> str:
    if float(x).is_integer() and abs(x) < 1000:
        return f"{x:.0f}"
    if abs(x) >= 1000:
        return f"{x / 1000:.{max(decimals - 2, 0)}f}k"
    if abs(x) >= 10:
        return f"{x:.{max(decimals - 2, 0)}f}"
    return f"{x:.{decimals}f}"


def edge_labels(edges) -> list[str]:
    """Shortest labels that keep every bin edge distinct."""
    for d in range(2, 6):
        labels = [fmt_edge(e, d) for e in edges]
        if len(set(labels)) == len(labels):
            return labels
    return [f"{e:g}" for e in edges]


def bin_feature(s: pd.Series, col: str, missing_label: str) -> pd.Series:
    """Ordered categorical of bins; NaN becomes its own last bin."""
    if col in COUNT_CAPS:
        cap = COUNT_CAPS[col]
        labels = [str(i) for i in range(cap)] + [f"{cap}+"]
        b = pd.cut(s.clip(upper=cap), bins=np.arange(-0.5, cap + 1), labels=labels)
    else:
        b, edges = pd.qcut(s, q=10, duplicates="drop", retbins=True)
        labels = edge_labels(edges[1:])
        b = b.cat.rename_categories(labels)
    if s.isna().any():
        b = b.cat.add_categories(missing_label).fillna(missing_label)
    return b


# ---- Analysis (language independent) ----

def analyze(df: pd.DataFrame, target: str, monotone: dict) -> dict:
    y = df[target]
    out: dict = {"rows": len(df), "default_rate": round(float(y.mean()), 4)}

    out["target_corr"] = {
        c: {
            "pearson": round(float(df[c].corr(y)), 4),
            "spearman": round(float(df[c].corr(y, method="spearman")), 4),
        }
        for c in FEATURES + FLAGS
    }

    # Direction check: does the observed trend match the YAML monotone sign?
    checks = {}
    for c in FEATURES:
        rates = df.groupby(bin_feature(df[c], c, "missing"), observed=True)[target].mean()
        rates = rates.drop("missing", errors="ignore")
        steps = np.sign(np.diff(rates.values))
        rho = out["target_corr"][c]["spearman"]
        expected = monotone.get(c, 0)
        checks[c] = {
            "yaml_monotone": expected,
            "spearman_sign": int(np.sign(rho)),
            "binned_rate_pct": [round(float(r) * 100, 2) for r in rates.values],
            # Steps between adjacent bins that go against the overall trend
            "reversals": int((steps == -np.sign(rho)).sum()),
            "consistent": bool(expected == 0 or np.sign(rho) == expected),
        }
    out["direction_check"] = checks

    out["missing_pct"] = {
        c: round(float(v) * 100, 2) for c, v in df[FEATURES].isna().mean().items() if v > 0
    }
    thin = (df["NumberOfOpenCreditLinesAndLoans"] <= 2) & (df["NumberRealEstateLoansOrLines"] == 0)
    out["thin_filer"] = {
        "share": round(float(thin.mean()), 4),
        "rows": int(thin.sum()),
        "default_rate_thin": round(float(y[thin].mean()), 4),
        "default_rate_others": round(float(y[~thin].mean()), 4),
    }
    spearman = df[FEATURES].corr(method="spearman")
    pairs = [
        (a, b, round(float(spearman.loc[a, b]), 3))
        for i, a in enumerate(FEATURES) for b in FEATURES[i + 1:]
        if abs(spearman.loc[a, b]) >= 0.4
    ]
    out["high_feature_corr_pairs"] = pairs
    return out


# ---- Figures ----

def fig_class_balance(df, target, lang):
    T, rate = TEXT[lang], df[target].mean()
    fig, ax = plt.subplots(figsize=(6, 2.2))
    vals = [(1 - rate) * 100, rate * 100]
    counts = [(df[target] == 0).sum(), (df[target] == 1).sum()]
    ax.barh([T["non_default"], T["default"]], vals, color=[BLUE, ORANGE], height=0.55)
    for i, (v, n) in enumerate(zip(vals, counts)):
        ax.text(v + 1, i, f"{v:.1f}%  ({n:,} {T['rows']})", va="center", color=INK_2)
    ax.set_xlim(0, 115)
    ax.set_xlabel(T["share_rows"])
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_title(T["class_title"], loc="left")
    save(fig, "01_class_balance", lang)


def fig_default_by_bin(df, target, lang):
    T, S = TEXT[lang], SHORT[lang]
    overall = df[target].mean() * 100
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharey=False)
    for ax, c in zip(axes.flat, FEATURES):
        b = bin_feature(df[c], c, T["missing"])
        rates = df.groupby(b, observed=True)[target].mean() * 100
        colors = [GRAY_BAR if k == T["missing"] else BLUE for k in rates.index]
        ax.bar(range(len(rates)), rates.values, color=colors, width=0.8)
        ax.axhline(overall, color=INK_2, lw=0.9, ls="--")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index, rotation=45, ha="right", fontsize=7)
        ax.set_title(S[c], loc="left")
        ax.set_xlabel(T["bin_x"], fontsize=7, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel(T["default_rate"])
    axes[0, 0].annotate(f"{T['overall']} {overall:.1f}%", xy=(0.02, overall), xycoords=("axes fraction", "data"),
                        xytext=(0, 4), textcoords="offset points", fontsize=7, color=INK_2)
    fig.suptitle(T["bin_title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "02_default_rate_by_bin", lang)


def fig_distributions(df, target, lang):
    T, S = TEXT[lang], SHORT[lang]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, c in zip(axes, SKEWED):
        # Utilization is bounded after cleaning, so it reads best on a linear axis
        log = c != "RevolvingUtilizationOfUnsecuredLines"
        tf = (lambda v: np.log10(v + 1)) if log else (lambda v: v)
        hi = tf(df[c].quantile(0.99)) if log else 1.5
        bins = np.linspace(0, hi, 50)
        for val, color, label in [(0, BLUE, T["non_default"]), (1, ORANGE, T["default"])]:
            xv = tf(df.loc[df[target] == val, c].dropna())
            w = np.full(len(xv), 100 / len(xv))
            ax.hist(xv.clip(upper=hi), bins=bins, weights=w, histtype="step", lw=2, color=color, label=label)
        if log:
            ticks = [v for v in LOG_TICKS[c] if np.log10(v + 1) <= hi]
            ax.set_xticks([np.log10(v + 1) for v in ticks], [fmt_edge(v) for v in ticks])
        ax.set_title(S[c], loc="left")
        ax.set_xlabel(T["log_x"] if log else T["linear_x"], fontsize=8)
    axes[0].set_ylabel(T["density"])
    axes[0].legend(loc="upper center")
    fig.suptitle(T["dist_title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "03_distributions_log", lang)


def fig_corr_heatmap(df, lang):
    T, S = TEXT[lang], SHORT[lang]
    corr = df[FEATURES].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    labels = [S[c] for c in FEATURES]
    ax.set_xticks(range(len(FEATURES)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES)), labels)
    ax.grid(False)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(T["corr_title"], loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "04_feature_correlation", lang)


def fig_missing_flags(df, target, lang):
    T, S = TEXT[lang], SHORT[lang]
    miss = (df[FEATURES].isna().mean() * 100).loc[lambda s: s > 0].sort_values()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    a1.barh([S[c] for c in miss.index], miss.values, color=BLUE, height=0.6)
    for i, v in enumerate(miss.values):
        a1.text(v + 0.3, i, f"{v:.2f}%", va="center", color=INK_2, fontsize=8)
    a1.set_xlabel(T["missing_pct"]); a1.set_xlim(0, miss.max() * 1.2)
    a1.grid(axis="x"); a1.grid(axis="y", visible=False)
    a1.set_title(T["missing_title"], loc="left")

    x = np.arange(len(FLAGS)); w = 0.38
    r0 = [df.loc[df[f] == 0, target].mean() * 100 for f in FLAGS]
    r1 = [df.loc[df[f] == 1, target].mean() * 100 for f in FLAGS]
    a2.bar(x - w / 2 - 0.01, r0, w, color=BLUE, label=T["flag0"])
    a2.bar(x + w / 2 + 0.01, r1, w, color=ORANGE, label=T["flag1"])
    for xi, v in zip(x + w / 2, r1):
        a2.text(xi, v + 1, f"{v:.1f}%", ha="center", color=INK_2, fontsize=8)
    a2.set_xticks(x, [S[f] for f in FLAGS])
    a2.set_ylabel(T["default_rate"]); a2.legend(loc="upper right")
    a2.set_title(T["flag_title"], loc="left")
    fig.tight_layout()
    save(fig, "05_missing_and_flags", lang)


def fig_target_corr(summary, lang):
    T, S = TEXT[lang], SHORT[lang]
    r = pd.Series({c: v["pearson"] for c, v in summary["target_corr"].items() if c in FEATURES})
    r = r.reindex(r.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(0.3, 0.5, color=ORANGE, alpha=0.12, lw=0)
    ax.text(0.4, -0.45, T["target_band"], ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.barh([S[c] for c in r.index], r.abs().values, color=BLUE, height=0.6)
    for i, v in enumerate(r.values):
        ax.text(abs(v) + 0.008, i, f"{v:+.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, 0.55); ax.set_xlabel(T["abs_r"])
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title(T["target_title"], loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "06_target_correlation", lang)


def fig_thin_filer(summary, lang):
    T, t = TEXT[lang], summary["thin_filer"]
    groups = [T["thin"], T["not_thin"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.6))
    for ax, vals, title in [
        (a1, [t["share"] * 100, (1 - t["share"]) * 100], T["thin_title_size"]),
        (a2, [t["default_rate_thin"] * 100, t["default_rate_others"] * 100], T["thin_title_rate"]),
    ]:
        ax.barh(groups, vals, color=[ORANGE, BLUE], height=0.55)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, f"{v:.2f}%", va="center", color=INK_2, fontsize=8)
        ax.set_xlim(0, max(vals) * 1.25); ax.invert_yaxis()
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    a1.set_xlabel(T["share_rows"]); a2.set_xlabel(T["default_rate"])
    fig.tight_layout()
    save(fig, "07_thin_filer_preview", lang)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml", encoding="utf-8"))
    act = yaml.safe_load(open(ROOT / "config" / "actionability.yaml", encoding="utf-8"))
    monotone = {k: v["monotone"] for k, v in act["features"].items()}
    target = cfg["target"]
    df = pd.read_csv(ROOT / cfg["processed_path"])

    summary = analyze(df, target, monotone)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "en"):
        set_style(lang)
        fig_class_balance(df, target, lang)
        fig_default_by_bin(df, target, lang)
        fig_distributions(df, target, lang)
        fig_corr_heatmap(df, lang)
        fig_missing_flags(df, target, lang)
        fig_target_corr(summary, lang)
        fig_thin_filer(summary, lang)

    (ROOT / "reports" / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved figures to {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
