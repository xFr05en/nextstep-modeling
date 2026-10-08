# Model Report

> Educational use only. Not for real financial decisions. The alternative variables are simulated from the real outcome for every row, including the test set (as in the mission's example), so results with alternative data are optimistic. The GMSC-only model is the realistic reference.

- Code: `src/train.py` (run `python -m src.train --stage all`, about 7 minutes), settings in `config/train.yaml`, scoring in `config/scoring.yaml`
- Figures and SHAP: `python notebooks/model_report.py` (figures 11 and 12, `reports/shap_global_top10.csv`)
- Results: `reports/final_summary.json` (official), `reports/model_comparison.csv`, `reports/monotonic_results.csv`, `reports/grade_table.csv`, `reports/cv_fold_auc.csv`
- Model: `models/xgboost_v1.0.joblib`. MLflow guide: `reports/mlflow_schema_en.md`

## 1. Setup

| Part | Rows | Used for |
|---|---|---|
| Train (70%) | 104,999 | model comparison: 27 runs, 3 × 5-fold stratified CV (target × thin-filer), the winner rule |
| Validation (15%) | 22,500 | decisions: monotonic AUC loss (and any future threshold or fairness tuning) |
| Test (15%) | 22,500 | touched once, by the shipped model trained on train + validation |

- The split is `stratify=y` (default rate 6.68% in every part) and saved as `data/processed/split.csv` for the whole team.
- Imputation (median), LR transforms, scaling and SMOTE are pipeline steps, fit on training folds only.
- 21 features: 14 GMSC (10 original + 4 cleaning flags) and 7 alternative variables. `autopay_ratio` and `gender_female` are in the data but not model inputs.
- Fixed hyperparameters (no per-run tuning). The score-475 approval cutoff is a fixed rule from `config/scoring.yaml`; no test data sets any threshold.

## 2. Model comparison (CV on train, 15 folds; Figure 11)

| Model | Imbalance | GMSC only | Alternative only | Both | Thin-filer gain | Brier (both) | F1 (both) |
|---|---|---|---|---|---|---|---|
| LR | none | 0.850 | 0.914 | 0.939 | +0.103 | 0.037 | 0.514 |
| LR | class weights | 0.855 | 0.915 | 0.940 | +0.099 | 0.098 | 0.235 |
| LR | SMOTE | 0.855 | 0.914 | 0.939 | +0.100 | 0.096 | 0.245 |
| LightGBM | none | 0.862 | 0.911 | 0.942 | +0.090 | 0.036 | 0.535 |
| LightGBM | class weights | 0.861 | 0.911 | 0.941 | +0.090 | 0.082 | 0.286 |
| LightGBM | SMOTE | 0.854 | 0.910 | 0.942 | +0.107 | 0.036 | 0.508 |
| **XGBoost** | **none** | **0.864** | **0.913** | **0.942** | **+0.090** | **0.036** | **0.533** |
| XGBoost | class weights | 0.862 | 0.912 | 0.942 | +0.091 | 0.087 | 0.260 |
| XGBoost | SMOTE | 0.851 | 0.911 | 0.941 | +0.109 | 0.037 | 0.468 |

AUC is the mean over 15 folds (SD 0.002 to 0.003 for every "both" run).

- **Adding alternative data matters far more than the model:** about +0.08 AUC; the three algorithms differ by at most 0.003.
- **Imbalance handling does not help AUC.** Class weights inflate the predicted probabilities (Brier 0.036 → 0.087) and lower F1 at the fixed cutoff, because more applicants fall below it.
- **The larger SMOTE gain is not a better model:** SMOTE lowers the GMSC-only baseline more.
- **"Alternative only" beats "GMSC only"** because of the simulation settings, not real data.
- **Winner rule:** highest CV AUC among "both" runs, ties within 1 SD resolved toward no resampling. **Result: XGBoost, no resampling.**

## 3. Monotonic constraints (validation)

13 constraints from the YAML `monotone` field: +1 for utilization, the three late-payment counts and debt ratio; −1 for income and the 7 alternative variables. Age, open lines, real estate loans, dependents and the 4 flags are unconstrained.

| Model | Validation AUC unconstrained | constrained | Loss (95% paired bootstrap CI) |
|---|---|---|---|
| XGBoost | 0.9438 | 0.9438 | **0.0000** (−0.0012 to +0.0011) |
| LightGBM | 0.9430 | 0.9436 | −0.0007 (−0.0020 to +0.0006) |

No constraint is dropped (limit 0.01). The shipped model makes 0 wrong-direction moves on all 13 constrained features (tested).

## 4. Official results: test set (shipped model, trained on train + validation)

95% intervals from 500 bootstrap resamples of the test rows.

| Target | Test result | 95% CI | Pass |
|---|---|---|---|
| AUC ≥ 0.78 | **0.949** | 0.944 to 0.953 | yes |
| KS ≥ 0.28 | **0.760** | 0.745 to 0.776 | yes |
| Thin-filer AUC gain ≥ +0.03 | **+0.071** | +0.052 to +0.092 (paired bootstrap) | yes |
| PSI (train + validation vs. test) < 0.1 | **0.0004** | | yes |
| Monotonic AUC loss ≤ 0.01 | **0.0000** (validation) | −0.0012 to +0.0011 | yes |

- **Thin-filer gain:** thin-filer AUC 0.932 with alternative data vs. 0.861 for the same setup on GMSC features only (1,761 thin-filers, 221 defaults in the test set). The CV estimate on train was +0.090; the lower test value is the usual optimism of CV plus the small number of thin-filer defaults.
- **Precision / recall / F1 at the score-475 cutoff** (positive class = default, predicted default = rejected): 0.414 / 0.810 / **0.548** (F1 CI 0.529 to 0.566).
- **Calibration:** mean predicted PD 6.60% vs. actual 6.68%, Brier 0.035. No calibration step was needed (no resampling).
- **PSI note:** train and test are random samples of the same population, so PSI is close to 0 by construction; it is not a test of drift over time (GMSC has no dates).

## 5. What drives the model: global SHAP top 10 (test set)

| # | Feature | Mean \|SHAP\| | Class |
|---|---|---|---|
| 1 | app_login_frequency | 0.671 | NOT_RECOMMENDED |
| 2 | regular_payment_count | 0.553 | NOT_RECOMMENDED |
| 3 | spending_consistency | 0.528 | ACTIONABLE |
| 4 | telecom_tenure_months | 0.469 | TIME_ONLY |
| 5 | insurance_paid_months | 0.452 | TIME_ONLY |
| 6 | RevolvingUtilizationOfUnsecuredLines | 0.415 | ACTIONABLE |
| 7 | NumberOfTimes90DaysLate | 0.216 | NON_DECREASING |
| 8 | age | 0.214 | IMMUTABLE |
| 9 | NumberOfTime30-59DaysPastDueNotWorse | 0.204 | NON_DECREASING |
| 10 | telecom_payment_rate | 0.152 | ACTIONABLE |

- **The two NOT_RECOMMENDED count variables lead.** They carry slightly more of the mission variables' shared signal (latent r 0.56 vs. 0.47 for the payment rates), so the trees take the common signal mostly from them. In rejection reasons they must appear as "reference (not changeable)", never as advice.
- **`utility_payment_rate` is outside the top 10**, and `telecom_payment_rate` is only #10, for the same reason: the shared factor lets the model take the common signal from the count variables.

## 6. Score, grades and approval (test set, Figure 12)

Score = 600 + (50 / ln 2) × ln(odds / 50), clipped to 0 to 1000. Approval: grades A to C (score ≥ 475). Recourse target: 495 (`recourse_target_score`, 20 points above the cutoff so paths survive retraining).

| Grade | Score | Share (all) | Share (thin-filers) | Mean predicted PD | Actual default rate |
|---|---|---|---|---|---|
| A | 600+ | 65.7% | 53.2% | 0.5% | 0.3% |
| B | 530 to 599 | 13.9% | 12.8% | 3.2% | 2.6% |
| C | 475 to 529 | 7.3% | 8.7% | 7.2% | 9.3% |
| D | 420 to 474 | 4.3% | 6.2% | 14.2% | 15.4% |
| E | below 420 | 8.8% | 19.1% | 53.3% | 54.0% |

Approval rate: 86.9% overall, **74.7% for thin-filers**, 88.0% for others. Default rate 1.5% among approved, 41.4% among declined.

## 7. MLflow and the model file

- One experiment per algorithm (`nextstep-lr`, `nextstep-lgbm`, `nextstep-xgb`), plus `nextstep-final` and `nextstep-audit`. Every run logs parameters, AUC, KS and F1; the best run of each algorithm and the final run hold a model artifact with signature and input example.
- The shipped model is saved as `models/xgboost_v1.0.joblib` (version in `config/train.yaml`). Model Registry: 윤제진.

## 8. Limitations

- **Simulation bias:** the simulator uses the target for all rows, test included. The realistic reference is GMSC only: CV AUC 0.864 on train, thin-filer test AUC 0.861.
- **Strength set by the mission rule:** each mission variable alone reaches AUC 0.80 to 0.84.
- One seed and fixed hyperparameters. Grade cutoffs, the approval rule and recourse costs are team assumptions, not values from a lender.
