"""Figures for the Step 5 model report. Reads the CSV/JSON outputs of `python -m src.train`.

Writes reports/figures/11_model_comparison_{ko,en}.png and 12_grades_{ko,en}.png
Run from the project root:  python notebooks/model_report.py
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

from eda import BLUE, INK_2, ORANGE, save, set_style  # noqa: E402

AQUA = "#1baf7a"  # categorical slot 3
RES_COLORS = {"none": BLUE, "class_weight": ORANGE, "smote": AQUA}
MODELS = {"lr": "LR", "lgbm": "LightGBM", "xgb": "XGBoost"}
TXT = {
    "en": {
        "cmp_title": "Cross-validated AUC by model, feature set and imbalance handling",
        "fs": {"gmsc": "GMSC only", "alt": "Alternative only", "both": "Both"},
        "res": {"none": "None", "class_weight": "Class weights", "smote": "SMOTE"},
        "auc": "AUC (mean of 5 folds)", "note": "Error bars: plus or minus 1 SD across 5 folds. All runs are above the charter minimum AUC of 0.78.",
        "grade_title_share": "Share of borrowers by grade", "grade_title_rate": "Actual default rate vs. predicted PD by grade",
        "all": "All borrowers", "thin": "Thin-filers", "actual": "Actual default rate", "pred": "Mean predicted PD",
        "share": "Share (%)", "rate": "Rate (%)", "approved": "Left of line: approved (A to C)",
    },
    "ko": {
        "cmp_title": "모델, 변수 집합, 불균형 처리별 교차검증 AUC",
        "fs": {"gmsc": "GMSC만", "alt": "대체 데이터만", "both": "둘 다"},
        "res": {"none": "없음", "class_weight": "클래스 가중치", "smote": "SMOTE"},
        "auc": "AUC (5개 폴드 평균)", "note": "오차 막대: 5개 폴드의 표준편차 1배. 모든 실행이 헌장 최소 AUC 0.78보다 높음.",
        "grade_title_share": "등급별 대출자 비율", "grade_title_rate": "등급별 실제 부도율과 예측 PD",
        "all": "전체 대출자", "thin": "씬파일러", "actual": "실제 부도율", "pred": "평균 예측 PD",
        "share": "비율 (%)", "rate": "비율 (%)", "approved": "선 왼쪽: 승인 (A~C)",
    },
}


def fig_comparison(cmp: pd.DataFrame, lang: str) -> None:
    T = TXT[lang]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, fs in zip(axes, ["gmsc", "alt", "both"]):
        sub = cmp[cmp["features"] == fs]
        for j, (res, color) in enumerate(RES_COLORS.items()):
            r = sub[sub["resampling"] == res].set_index("model").loc[list(MODELS)]
            x = np.arange(len(MODELS)) + (j - 1) * 0.22
            ax.errorbar(x, r["auc_mean"], yerr=r["auc_std"], fmt="o", ms=8, color=color, capsize=3,
                        label=T["res"][res])
        ax.set_xticks(range(len(MODELS)), MODELS.values())
        ax.set_title(T["fs"][fs], loc="left")
        ax.set_xlim(-0.6, 2.6)
    axes[0].set_ylabel(T["auc"])
    axes[-1].legend(loc="lower right")
    fig.suptitle(T["cmp_title"], x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.02, T["note"], fontsize=7.5, color=INK_2)
    fig.tight_layout()
    save(fig, "11_model_comparison", lang)


def fig_grades(gt: pd.DataFrame, lang: str) -> None:
    T = TXT[lang]
    g = gt.index.tolist()
    x = np.arange(len(g))
    w = 0.38
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (c1, l1), (c2, l2), ylabel in [
        (a1, ("share", T["all"]), ("thin_share", T["thin"]), T["share"]),
        (a2, ("default_rate", T["actual"]), ("mean_pd", T["pred"]), T["rate"]),
    ]:
        v1, v2 = gt[c1] * 100, gt[c2] * 100
        ax.bar(x - w / 2 - 0.01, v1, w, color=BLUE, label=l1)
        ax.bar(x + w / 2 + 0.01, v2, w, color=ORANGE, label=l2)
        for xi, a, b in zip(x, v1, v2):
            ax.text(xi - w / 2, a + 0.8, f"{a:.1f}", ha="center", fontsize=7.5, color=INK_2)
            ax.text(xi + w / 2, b + 0.8, f"{b:.1f}", ha="center", fontsize=7.5, color=INK_2)
        ax.set_ylim(0, max(v1.max(), v2.max()) * 1.18)
        ax.axvline(2.5, color=INK_2, lw=0.9, ls="--")
        ax.text(2.58, ax.get_ylim()[1] * 0.97, T["approved"], ha="left", va="top", fontsize=8, color=INK_2)
        ax.set_xticks(x, g)
        ax.set_ylabel(ylabel)
        ax.legend(loc="center right" if c1 == "share" else "upper left")
    a1.set_title(T["grade_title_share"], loc="left")
    a2.set_title(T["grade_title_rate"], loc="left")
    fig.tight_layout()
    save(fig, "12_grades", lang)


def main() -> None:
    cmp = pd.read_csv(ROOT / "reports" / "model_comparison.csv")
    gt = pd.read_csv(ROOT / "reports" / "grade_table.csv", index_col="grade")
    json.loads((ROOT / "reports" / "final_summary.json").read_text())  # fail early if missing
    for lang in ("ko", "en"):
        set_style(lang)
        fig_comparison(cmp, lang)
        fig_grades(gt, lang)
    print("Saved figures 11 and 12")


if __name__ == "__main__":
    main()
