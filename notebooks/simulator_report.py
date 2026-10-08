"""Simulator validation figures and credit-link trade-off sweep (Step 3).

Needs data/processed/gmsc_clean.csv. Writes:
  reports/simulator_sweep.json
  reports/figures/08_thin_lift_sweep_{ko,en}.png
  reports/figures/09_alt_correlations_{ko,en}.png
  reports/figures/10_alt_distributions_{ko,en}.png

Run from the project root:  python notebooks/simulator_report.py   (about 2 minutes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from eda import BLUE, DIVERGING, INK, INK_2, ORANGE, SHORT, TEXT, save, set_style  # noqa: E402
from src.evaluate import thin_filer_auc_lift  # noqa: E402
from src.features import add_thin_filer_flag, feature_sets  # noqa: E402
from src.data.simulator import credit_score, load_config, simulate  # noqa: E402

ALT_SHORT = {
    "en": {
        "telecom_payment_rate": "Phone bill payment rate", "utility_payment_rate": "Utility bill payment rate",
        "telecom_tenure_months": "Phone carrier tenure (months)", "insurance_paid_months": "Insurance paid months",
        "autopay_ratio": "Autopay share", "spending_consistency": "Spending consistency",
        "regular_payment_count": "Recurring payments", "app_login_frequency": "App logins",
    },
    "ko": {
        "telecom_payment_rate": "통신비 정상납부율", "utility_payment_rate": "공과금 납부율",
        "telecom_tenure_months": "통신사 가입 기간(개월)", "insurance_paid_months": "건강보험·연금 납부 개월",
        "autopay_ratio": "자동이체 비율", "spending_consistency": "소비 일관성 점수",
        "regular_payment_count": "정기결제 건수", "app_login_frequency": "앱 로그인 빈도",
    },
}
TXT = {
    "en": {
        "sweep_title": "AUC gain from alternative data vs. link to credit behavior",
        "sweep_x": "Credit link strength b (0 = depends on default only)",
        "sweep_y": "AUC gain (with minus without alternative data)",
        "thin": "Thin-filers", "overall": "All borrowers", "goal": "Goal +0.03 (thin-filers)",
        "main": "main dataset",
        "sweep_note": "Error bars: plus or minus 1 SD across 5 CV folds. LightGBM default settings. Latent r with default fixed at 0.35 in every setting.",
        "corr_title": "Pearson r of simulated variables (main dataset, b = 0.30)",
        "default": "Default", "credit": "Credit score", "pastdue": "Past-due total",
        "util": "Utilization", "age": "Age",
        "dist_title": "Simulated variables by default status (main dataset)",
        "share": "Share of group (%)",
    },
    "ko": {
        "sweep_title": "대체 데이터의 AUC 향상 폭과 신용 행동 연결 강도",
        "sweep_x": "신용 연결 강도 b (0 = 부도에만 의존)",
        "sweep_y": "AUC 향상 (대체 데이터 포함 - 미포함)",
        "thin": "씬파일러", "overall": "전체 대출자", "goal": "목표 +0.03 (씬파일러)",
        "main": "주 데이터셋",
        "sweep_note": "오차 막대: 5개 CV 폴드의 표준편차 1배. LightGBM 기본 설정. 모든 설정에서 부도와의 잠재 상관은 0.35로 고정.",
        "corr_title": "시뮬레이션 변수의 Pearson r (주 데이터셋, b = 0.30)",
        "default": "부도", "credit": "신용 점수", "pastdue": "연체 횟수 합계",
        "util": "리볼빙 사용률", "age": "연령",
        "dist_title": "부도 여부별 시뮬레이션 변수 분포 (주 데이터셋)",
        "share": "집단 내 비율 (%)",
    },
}


def run_sweep(df, cfg, target, feats) -> list[dict]:
    results = []
    for b in cfg["credit_link"]["sweep"]:
        sim, info = simulate(df, cfg, b=b, target=target)
        lift = thin_filer_auc_lift(sim, target, feats["base"], feats["alt"], seed=cfg["seed"])
        results.append({"b": b, "noise": {k: v["noise"] for k, v in info["params"].items()},
                        "corr_target_latent": info["corr_target_latent"],
                        "corr_target": info["corr_target"], "auc_single": info["auc_single"], **lift})
        print(f"b={b}: thin lift {lift['thin_lift']['mean']:+.4f} +/- {lift['thin_lift']['std']:.4f}, "
              f"overall lift {lift['lift']['mean']:+.4f}")
    return results


def fig_sweep(results, main_b, lang):
    T = TXT[lang]
    b = np.array([r["b"] for r in results])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(main_b - 0.012, main_b + 0.012, color=AXIS_LIGHT, lw=0, zorder=0)
    ax.axhline(0.03, color=INK_2, lw=0.9, ls="--")
    ax.text(b[0], 0.03, T["goal"], ha="left", va="bottom", fontsize=8, color=INK_2)
    # Thin-filer labels above the error bar, overall labels below it
    for key, color, label, side in [("thin_lift", ORANGE, T["thin"], 1), ("lift", BLUE, T["overall"], -1)]:
        m = np.array([r[key]["mean"] for r in results])
        s_ = np.array([r[key]["std"] for r in results])
        ax.errorbar(b, m, yerr=s_, color=color, lw=2, marker="o", ms=7, capsize=3, label=label)
        for xi, yi, si in zip(b, m, s_):
            ax.annotate(f"{yi:+.3f}", (xi, yi + side * si), xytext=(0, 5 * side), textcoords="offset points",
                        ha="center", va="bottom" if side > 0 else "top", fontsize=8, color=INK_2)
    ax.set_ylim(0, 0.1)
    ax.text(main_b, 0.098, T["main"], ha="center", va="top", fontsize=8, color=INK_2)
    ax.set_xticks(b)
    ax.set_xlabel(T["sweep_x"]); ax.set_ylabel(T["sweep_y"])
    ax.legend(loc="lower right")
    ax.set_title(T["sweep_title"], loc="left", fontsize=12)
    fig.text(0.01, -0.01, T["sweep_note"], fontsize=7.5, color=INK_2, ha="left", va="top")
    fig.tight_layout()
    save(fig, "08_thin_lift_sweep", lang)


AXIS_LIGHT = "#e1e0d9"


def fig_corr(sim, cfg, target, lang):
    T, A = TXT[lang], ALT_SHORT[lang]
    names = list(cfg["variables"])
    cl = cfg["credit_link"]
    ref = pd.DataFrame({
        T["default"]: sim[target],
        T["credit"]: credit_score(sim, cfg),
        T["pastdue"]: sim[cl["pastdue_columns"]].sum(axis=1, min_count=1),
        T["util"]: sim[cl["util_column"]],
        T["age"]: sim["age"],
    })
    ref = pd.concat([ref, sim[names].rename(columns=A)], axis=1)
    corr = ref.corr().loc[[A[n] for n in names]]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(corr.shape[1]), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(corr.shape[0]), corr.index)
    ax.grid(False)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axvline(4.5, color="white", lw=3)  # separates reference columns from alternative variables
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(T["corr_title"], loc="left", fontsize=12)
    fig.tight_layout()
    save(fig, "09_alt_correlations", lang)


def fig_dist(sim, cfg, target, lang):
    T, A, S = TXT[lang], ALT_SHORT[lang], TEXT[lang]
    names = list(cfg["variables"])
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, n in zip(axes, names):
        x = sim[n]
        vals = np.sort(x.unique())
        if len(vals) <= 30:  # discrete: side-by-side bars
            w = (vals[1] - vals[0]) * 0.4 if len(vals) > 1 else 0.4
            for k, (val, color, label) in enumerate([(0, BLUE, S["non_default"]), (1, ORANGE, S["default"])]):
                share = x[sim[target] == val].value_counts(normalize=True).reindex(vals, fill_value=0) * 100
                ax.bar(vals + (k - 0.5) * w * 1.05, share.values, width=w, color=color, label=label)
        else:  # continuous: step histograms
            bins = np.linspace(0, x.quantile(0.99), 40)
            for val, color, label in [(0, BLUE, S["non_default"]), (1, ORANGE, S["default"])]:
                xv = x[sim[target] == val]
                ax.hist(xv.clip(upper=bins[-1]), bins=bins, weights=np.full(len(xv), 100 / len(xv)),
                        histtype="step", lw=2, color=color, label=label)
        ax.set_title(A[n], loc="left")
    axes[0].set_ylabel(T["share"])
    axes[0].legend(loc="upper left")
    fig.suptitle(T["dist_title"], x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "10_alt_distributions", lang)


def main() -> None:
    cfg = load_config()
    data_cfg = yaml.safe_load(open(ROOT / "config" / "data.yaml", encoding="utf-8"))
    target = data_cfg["target"]
    df = add_thin_filer_flag(pd.read_csv(ROOT / data_cfg["processed_path"]), cfg["thin_filer"]["rule"])
    feats = feature_sets()

    results = run_sweep(df, cfg, target, feats)
    (ROOT / "reports" / "simulator_sweep.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    sim, _ = simulate(df, cfg, target=target)
    for lang in ("ko", "en"):
        set_style(lang)
        fig_sweep(results, cfg["credit_link"]["b"], lang)
        fig_corr(sim, cfg, target, lang)
        fig_dist(sim, cfg, target, lang)
    print("Saved sweep and figures")


if __name__ == "__main__":
    main()
