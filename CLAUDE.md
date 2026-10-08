# NextStep: Data, Simulator, Modeling (Wonbin's part)

## Project context
- Course: Capstone Design II / Industry Project (CSE4187), Sogang University. Team 4무원 (4 members).
- System: XAI-based alternative credit scoring for thin-filers, with Actionable Recourse (approval paths built only from changes the applicant can make).
- This repo covers MY part only: data loading, alternative-data simulator, actionability metadata, monotonic-constraint model training.
- Teammates consume my outputs:
  - 채민규 (fairness, SHAP, DiCE recourse, cost function): needs `config/actionability.yaml` + trained model.
  - 윤제진 (MLflow registry, Docker Compose, FastAPI, Streamlit): will create the team repo in week 6. Keep this code drop-in ready.

## Data (place in `data/raw/`, never commit)
- GMSC: `cs-training.csv` (Kaggle "Give Me Some Credit"). Target: `SeriousDlqin2yrs`.
- German Credit: `german.data` (UCI) or Kaggle CSV. Secondary dataset.
- Known GMSC issues to handle: values 96/98 in the three past-due columns are special codes; `MonthlyIncome` about 20% missing; `NumberOfDependents` about 2.6% missing; `DebtRatio` holds raw amounts when income is missing; `RevolvingUtilizationOfUnsecuredLines` has values above 1; one row with age = 0. Verify all of these against the actual file before acting.

## Decisions (final, owned by Wonbin, last updated 2026-10-08, mission compliance)
- Goal: pass every item of the mission checklist (`../02_Project_Topic/37_XAI_대안신용평가_시스템_개발.pdf`, section 4) for my part.
- Alternative variables (8), defined in `config/simulator.yaml` and `config/actionability.yaml`:
  - Mission 5: telecom_payment_rate, utility_payment_rate, spending_consistency, regular_payment_count, app_login_frequency.
  - Extras 3: telecom_tenure_months, insurance_paid_months, autopay_ratio.
- Mission rule: the 5 mission variables are calibrated to OBSERVED Pearson r = -0.32 (+/- 0.005) with the 0/1 target (the checklist verifies df.corr()); extras stay at latent r = -0.35. Signs: all negative.
- Shared factor for the 5 mission variables, largest pair correlation <= 0.60. Count variables use Binomial(n, 0.75).
- app_login_frequency is NOT linked to age (protected-attribute proxy); regular_payment_count is NOT linked to income.
- Classes: spending_consistency ACTIONABLE; regular_payment_count, app_login_frequency, autopay_ratio NOT_RECOMMENDED (gaming risk). autopay_ratio is also not a model feature (model_feature: false).
- Generator: Gaussian copula (scipy). Seeded. Parameters in config, not hard-coded. The target is used for every row (as in the mission example); GMSC-only results are the realistic reference.
- Simulator parameters: thin_filer_ratio (default None, subsample mode; mask mode for demonstration only) and bias_ratio (default 0.0; shifts women and age 20-34 down by bias_ratio SD).
- Gender: simulated binary, independent of the target.
- Thin-filer: proxy NumberOfOpenCreditLinesAndLoans <= 2 AND NumberRealEstateLoansOrLines == 0 for every official number; the mission rule is implemented (is_thin_filer_mission) but flags 0% of raw GMSC.
- Split: stratify=y 70/15/15 (data/processed/split.csv). Model comparison: 3 x 5-fold CV inside train. Validation: monotonic loss and any tuning. Test: touched once by the shipped model (fit on train+validation). Official results are test-set figures with bootstrap CIs.
- Late-payment counts: "cannot decrease" (charter rule). Age: feature, no monotonic constraint (protected).
- Monotonic constraints: only from the `monotone` field in the YAML (13 constraints). Model output = probability of default.
- Scoring: approval at score >= 475 (fixed rule, scoring.yaml); recourse target 495.
- Model file: models/xgboost_v1.0.joblib (version in config/train.yaml). MLflow: one experiment per algorithm + final + audit; MLFLOW_TRACKING_URI.
- Open items for teammates: age-band EO gap 0.121 > 0.10 (채민규); grey zone around 475 (team); Model Registry (윤제진); ANOVA (채민규, data in reports/cv_fold_auc.csv).

## Required targets (from the team charter)
- AUC >= 0.78, KS >= 0.28, thin-filer AUC lift >= +0.03 (with vs. without alternative data), PSI < 0.1.
- AUC loss from monotonic constraints <= 0.01. If exceeded, reduce constrained variables.
- pytest: at least 10 tests, 80% passing.

## Work plan
1. Load and clean GMSC -> `data/processed/`. Short data report.
2. EDA: default rate, distributions, correlations. Save figures to `reports/figures/`.
3. Copula simulator -> add 5 alternative variables + gender. Validate correlations.
4. Thin-filer flag.
5. Model comparison: {LogisticRegression, LightGBM, XGBoost} x {GMSC only, alternative only, both} x {no resampling, class weights, SMOTE}, stratified CV. Then monotonic vs. unconstrained for the boosting models. Log every run to local MLflow (`mlruns/`).
6. pytest: simulator correlation range, reproducibility with seed, monotonicity of predictions, YAML schema, no target leakage.
7. README in Korean and English. Generate the actionability table Excel (Korean and English) directly from `config/actionability.yaml`, so the Excel and YAML never drift apart.

## Repo layout
```
config/      data, simulator, actionability, train, scoring, german (.yaml)
data/        raw/ (gitignored), processed/ (gitignored; split.csv defines the split)
src/data/    loader.py, preprocessor.py, simulator.py
src/         features.py, train.py, evaluate.py, scoring.py, actionability_table.py, german.py
tests/
models/      xgboost_v1.0.joblib
reports/     figures/, results tables, KO/EN reports, team handoff files
notebooks/   report, figure and audit scripts; no logic that src/ depends on
```

## Working rules
- Explain each step to me before moving on and wait for my review. I must be able to present this.
- Plain language, concise. No em dashes in documents.
- Every document deliverable: Korean and English versions. Code comments in English.
- Secrets in `.env`, never committed. Add a disclaimer: educational use only, not for real financial decisions.
