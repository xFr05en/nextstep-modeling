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

## Decisions (final, owned by Wonbin, last updated 2026-10-05)
- Alternative variables (5), defined in `config/actionability.yaml`:
  telecom_ontime_rate, utility_ontime_rate, telecom_tenure_months, insurance_paid_months, autopay_ratio.
- Mission rule: each alternative variable correlates with the target at |r| 0.3 to 0.5, measured as the copula's latent correlation (hidden score vs. hidden default score). All five set to 0.35. Observed Pearson and Spearman r are reported alongside. Signs: all negative (higher value = lower default).
- Generator: Gaussian copula (scipy). Seeded. Parameters in a config file, not hard-coded.
- Gender: simulated binary, independent of the target.
- Thin-filer proxy: NumberOfOpenCreditLinesAndLoans <= 2 AND NumberRealEstateLoansOrLines == 0. Report the share flagged.
- Late-payment counts: fixed as "cannot decrease" (charter rule).
- Age: used as a feature, no monotonic constraint (protected attribute).
- Monotonic constraints: only from the `monotone` field in the YAML. Model output = probability of default.

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
config/      actionability.yaml, simulator.yaml
data/        raw/ (gitignored), processed/ (gitignored)
src/         data.py, simulator.py, features.py, train.py, evaluate.py
tests/
models/
reports/     figures/, results tables
notebooks/   EDA only, no logic that src/ depends on
```

## Working rules
- Explain each step to me before moving on and wait for my review. I must be able to present this.
- Plain language, concise. No em dashes in documents.
- Every document deliverable: Korean and English versions. Code comments in English.
- Secrets in `.env`, never committed. Add a disclaimer: educational use only, not for real financial decisions.
