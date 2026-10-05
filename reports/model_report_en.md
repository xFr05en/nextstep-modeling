# Model Comparison Report (Step 5)

> Educational use only. Not for real financial decisions. Results use simulated alternative data (see `simulator_report_en.md`).

- Code: `src/train.py` (run `python -m src.train --stage all`, about 3 minutes), settings in `config/train.yaml`, scoring in `config/scoring.yaml`
- Figures: `python notebooks/model_report.py` (figures 11 and 12, Korean and English)
- Results: `reports/model_comparison.csv`, `reports/monotonic_results.csv`, `reports/final_summary.json`, `reports/grade_table.csv`
- MLflow: `mlruns/` (see `mlflow_schema_en.md`). Final model: `models/final_model.joblib`

## 1. Setup

- **Data:** `gmsc_sim.csv` (149,999 rows, main simulated dataset, b = 0.30).
- **Validation:** 5-fold CV, stratified by default and thin-filer flag, seed 42. Every run uses the same folds, so differences are paired.
- **Feature sets:** `gmsc` = 10 original features + 4 flags. `alt` = 5 simulated variables. `both` = all 19. Gender is never a feature.
- **No leakage:** median imputation, LR transforms (log1p, then cap at the training fold's 99th percentile), scaling, SMOTE and calibration are all steps of one pipeline. They are fit on the 4 training folds only and applied to the held-out fold. SMOTE never touches the held-out fold.
- **Fixed hyperparameters** in `config/train.yaml` (no per-run tuning), so the grid compares methods, not tuning effort.

## 2. Comparison grid (Figure 11)

AUC, mean ± SD over 5 folds:

| Model | Imbalance | GMSC only | Alternative only | Both | Thin-filer gain | Brier (both) |
|---|---|---|---|---|---|---|
| LR | none | 0.850 | 0.901 | 0.922 ± 0.003 | +0.081 | 0.041 |
| LR | class weights | 0.856 | 0.901 | 0.923 ± 0.003 | +0.077 | 0.110 |
| LR | SMOTE | 0.855 | 0.901 | 0.922 ± 0.003 | +0.078 | 0.108 |
| LightGBM | none | 0.864 | 0.899 | 0.926 ± 0.004 | +0.070 | 0.040 |
| LightGBM | class weights | 0.863 | 0.899 | 0.926 ± 0.004 | +0.070 | 0.096 |
| LightGBM | SMOTE | 0.855 | 0.894 | 0.926 ± 0.004 | +0.085 | 0.040 |
| **XGBoost** | **none** | **0.865** | **0.900** | **0.927 ± 0.004** | **+0.070** | **0.039** |
| XGBoost | class weights | 0.864 | 0.899 | 0.926 ± 0.004 | +0.070 | 0.099 |
| XGBoost | SMOTE | 0.853 | 0.895 | 0.924 ± 0.004 | +0.087 | 0.041 |

**What this shows:**
- **Adding alternative data matters far more than the choice of model.** Going from GMSC only to both adds about 0.06 to 0.07 AUC. The three models differ by at most 0.005, which is close to the fold-to-fold spread.
- **Imbalance handling does not help AUC.** SMOTE even lowers the GMSC-only tree models by about 0.01. Class weights roughly double or triple the Brier score (0.039 to 0.099 for XGBoost), because they push predicted probabilities up. That is a problem when the output must be a probability of default.
- **The larger thin-filer gain with SMOTE (+0.085) is not a better model.** SMOTE lowers the GMSC-only baseline more than the full model, so the difference grows.
- **"Alternative only" beats "GMSC only" (0.90 vs. 0.86).** This comes from the simulation settings, not from real data, and should not be presented as a finding about real alternative data.

## 3. Winner

Rule (from `config/train.yaml`): highest AUC among `both` runs. Runs within 1 fold SD of the best count as ties, and ties go to no resampling first.

**Result: XGBoost, both feature sets, no resampling** (AUC 0.9267 ± 0.0036). It is also the highest AUC, so the tie rule did not change the choice.

**Calibration:** not needed, because the winner uses no class weights or SMOTE (the Platt step is only added in that case). The probabilities are already well calibrated: mean predicted PD 6.66% vs. actual default rate 6.68%, Brier 0.0394. For comparison, the class-weight version has Brier 0.099 before calibration, which is why calibration would have been required there.

## 4. Monotonic constraints

Constraints come only from the YAML `monotone` field. All 11 nonzero entries were applied:

| Direction | Features |
|---|---|
| +1 (higher = more risk) | utilization, 30-59 / 60-89 / 90+ days late, debt ratio |
| -1 (higher = less risk) | monthly income, 5 alternative variables |
| 0 (none) | age (protected), open lines, real estate loans, dependents, 4 flags |

| Model | AUC unconstrained | AUC constrained | AUC loss | Limit |
|---|---|---|---|---|
| LightGBM | 0.92622 | 0.92621 | 0.00001 | 0.01 |
| XGBoost | 0.92674 | 0.92669 | 0.00005 | 0.01 |

- **The loss is far below 0.01, so no constraint was dropped.** Debt ratio, first in the drop order, keeps its constraint, and `actionability.yaml` is unchanged.
- **The constraints really work.** Moving each constrained feature over its range for 300 sample rows, the unconstrained models make about 4,700 wrong-direction steps; the constrained final model makes 0.
- **LR has no constraints, but all 11 coefficient signs agree with the YAML.**
- **Why so cheap:** the Step 2 direction check already showed that every constrained feature moves the right way in the data. The constraint mainly removes small noisy reversals.

## 5. Charter targets (final model)

| Target | Value | Pass |
|---|---|---|
| AUC ≥ 0.78 | 0.927 ± 0.004 | yes |
| KS ≥ 0.28 | 0.704 ± 0.012 | yes |
| Thin-filer AUC gain ≥ +0.03 | +0.071 ± 0.008 | yes |
| PSI < 0.1 | 0.0005 | yes |
| Monotonic AUC loss ≤ 0.01 | 0.0001 | yes |

The thin-filer gain is paired: the same final setup (XGBoost, constraints, no resampling) trained on GMSC features only reaches a thin-filer AUC of 0.847, and 0.917 with alternative data.

**About PSI:** train and test folds are random samples of the same population, so PSI is close to 0 by construction. It confirms that scores are stable between folds, but it is not a test of drift over time. GMSC has no date column, so a real drift test is not possible here.

## 6. Credit score, grades and approval (`config/scoring.yaml`, Figure 12)

**Score formula:** `score = 600 + (50 / ln 2) × ln(odds / 50)`, where `odds = (1 - PD) / PD`. Clipped to 0 to 1000, higher = safer.

Why this choice:
- **Points-to-double-odds scaling is the standard way to turn a PD into a scorecard score.** Every 50 points doubles the good:bad odds, so equal score gaps mean equal changes in risk.
- **The 0 to 1000 range matches the point scale Korean credit bureaus use,** which is easy for the audience to read.
- **Anchoring:** 600 = 50:1 odds (PD about 2%). The average borrower (PD 6.7%) lands near 508.

**Grades:** cutoffs follow PD bands (A: PD below about 2%, B: 2% to 5%, C: 5% to 10%, D: 10% to 20%, E: 20% and above).

| Grade | Score | Share (all) | Share (thin-filers) | Mean predicted PD | Actual default rate |
|---|---|---|---|---|---|
| A | 600+ | 61.1% | 46.1% | 0.6% | 0.5% |
| B | 530 to 599 | 16.3% | 15.6% | 3.2% | 3.1% |
| C | 475 to 529 | 8.4% | 9.8% | 7.2% | 8.0% |
| D | 420 to 474 | 5.3% | 8.3% | 14.1% | 16.0% |
| E | below 420 | 8.9% | 20.2% | 49.6% | 48.7% |

The actual default rate of each grade is close to its predicted PD, so the grades mean what they say. All numbers use out-of-fold predictions, never in-sample ones.

**Approval rule:** approve grades A to C (score 475 or higher, PD below about 10%).

| Group | Approval rate |
|---|---|
| All borrowers | 85.8% |
| Thin-filers | 71.5% |
| Others | 87.0% |

Default rate among approved borrowers: 1.7%. Among declined: 36.5%.

Thin-filers are approved less often because their default rate is about twice as high (12.9% vs. 6.2%). For 채민규's DiCE: paths must reach `recourse_target_score` = 495 in `config/scoring.yaml`, 20 points above the approval cutoff of 475, so that they stay approved when the model is retrained (decided after the audit, see `audit_report_en.md`).

## 7. Limitations

- **Simulation bias:** the alternative variables were generated from the real outcome, so the AUC with alternative data (0.927) and the gain (+0.071) are optimistic. The GMSC-only AUC (0.865) is the realistic reference.
- **PSI with random folds** is close to 0 by construction (see section 5).
- One seed, fixed hyperparameters. Fine for comparing methods. Tuning could add a little AUC but would not change the conclusions.
- Grade cutoffs and the approval rule are policy choices for this project, not values from a real lender.
