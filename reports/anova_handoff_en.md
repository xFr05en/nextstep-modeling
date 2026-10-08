# ANOVA Handoff (data for 체민규)

> Educational use only. Not for real financial decisions. Wonbin provides the data; 체민규 runs every ANOVA.

**File:** `reports/cv_fold_auc.csv` (from `python -m src.train --stage compare`). One row per run × repeat × fold × segment: 27 runs × 5 repeats × 5 folds × 3 segments = 2,025 rows (`cv_repeats: 5` in `config/train.yaml`).

| Column | Meaning |
|---|---|
| `run` | `{model}__{features}__{resampling}`, the same name as the MLflow run |
| `model` | `lr`, `xgb`, `lgbm` |
| `features` | `gmsc` (전통 Only), `alt` (대안 Only), `both` (통합) |
| `resampling` | `none`, `class_weight`, `smote` |
| `repeat`, `fold` | CV repeat 0 to 4 and fold 0 to 4 (25 replicates per run) |
| `segment` | `all`, `thin` (proxy thin-filer), `general` (not thin-filer) |
| `n_rows`, `n_defaults` | held-out rows and defaults in that fold and segment |
| `auc` | AUC on that fold's held-out rows within the segment |

All folds are 5 × 5-fold repeated stratified CV (target × thin-filer) inside the **train** split (`data/processed/split.csv`); every run uses the same folds, so rows are paired by `repeat` and `fold`.

**Which rows to use:**

| Test | Rows | Groups |
|---|---|---|
| Test 1 (required): one-way by features | `segment == "all"` | `features`: gmsc / alt / both |
| Test 2 (required): one-way by model | `segment == "all"` | `model`: lr / xgb / lgbm |
| Bonus: two-way | `segment in (thin, general)`, `features in (gmsc, both)` | segment × features (recommend the final setup, `model == "xgb"`, `resampling == "none"`) |

**Mission reporting rules:** report F, p-value and η² for each test; run Tukey HSD whenever p < 0.05; at least one test must reach p < 0.05 (test 1 will, by a wide margin); include at least one page of interpretation in the report.

**Power check (bonus two-way, final setup xgb/none, 25 per cell):** per-fold AUC SD 0.0072 to 0.0102 (thin) and 0.0031 to 0.0054 (general); thin-filer lift +0.0901 vs. general lift +0.0817, so the interaction is +0.0084. Approximate power at α = 0.05: **0.85** with 5 repeats (0.61 with 3 repeats, 0.24 to 0.38 with a single 5-fold run). Tests 1 and 2 have very large effects and are well powered either way.

**Limitations (state them in the report):**
1. **Fold AUCs are not independent samples.** The folds share most of their training data, so the ANOVA p-values are approximate (usually too small).
2. **In the two-way test, thin and general AUCs of a fold come from the same model** and are not independent either. Optional check: a paired test on the per-fold interaction contrast, or a mixed model with fold as a random effect.

**Team split rule:** `data/processed/split.csv` (row_id, split) defines train / validation / test for everyone. Tune fairness mitigation and thresholds on **validation**; report them on **test**. The score-475 approval cutoff is a fixed rule from `config/scoring.yaml`.
