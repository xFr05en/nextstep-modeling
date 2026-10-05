# German Credit Secondary Model (for the real-gender fairness check)

> Educational use only. Not for real financial decisions. This is a separate small model; the GMSC pipeline, configs and final model are unchanged.

**Data credit:** Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77. Licensed under CC BY 4.0.

- Code: `src/german.py` (run `python -m src.german`, about 20 seconds), settings and label mapping in `config/german.yaml`
- Data: download `german.data` from UCI into `data/raw/` (gitignored, not committed)
- Outputs: `reports/german/oof_predictions.csv` (for 채민규), `reports/german/metrics.json`
- Tests: `tests/test_german.py`

## 1. Data

- 1,000 applicants, 20 attributes, no missing values. Default rate 30%.
- All category codes (A11, A43, ...) are decoded into readable labels from UCI's documentation.
- Target recoded: **1 = default** (originally 2 = bad), 0 = good.
- **Sex** is derived from attribute 9 (personal status and sex): A92 = female, A91/A93/A94 = male. A95 (single female) does not occur in the data.

| Group | Rows | Default rate |
|---|---|---|
| Female (all "divorced/separated/married") | 310 | 35.2% |
| Male | 690 | 27.7% |
| Under 30 | 371 | 36.9% |
| 30 to 49 | 504 | 25.6% |
| 50+ | 125 | 27.2% |
| Foreign worker: yes | 963 | 30.7% |
| Foreign worker: no | 37 | 10.8% |

These are descriptive only. The fairness metrics are 채민규's work.

## 2. Features

- **Excluded from features, kept for evaluation:** attribute 9 (personal status and sex) and attribute 20 (foreign worker).
- **18 features:** 7 numeric (duration, credit amount, installment rate, years at residence, **age**, existing credits, people liable) and 11 categorical, one-hot encoded inside each training fold.
- **Age is a feature with no monotonic constraint**, the same approach as the GMSC model.

**Monotonic constraints:** only on numeric variables with a clear direction that the binned data confirms (default rate by quintile):

| Variable | Default rate by bin | Constraint |
|---|---|---|
| Duration (months) | 21%, 18%, 32%, 33%, 48% | +1 |
| Installment rate (% of income) | 25%, 27%, 29%, 33% | +1 |
| Credit amount | 30%, 24%, 27%, 26%, 43% (U-shaped) | none |
| Age | 39%, 32%, 26%, 26%, 25% | none (same as GMSC) |
| Residence, existing credits, people liable | no clear pattern | none |

Constraining credit amount as well would cost 0.008 AUC, against 0.0009 for the two constraints used.

## 3. Results

Logistic regression baseline and XGBoost (300 trees, depth 3, learning rate 0.05). Repeated stratified 5-fold CV, 10 repeats (50 fits per model). "± SD" is across the 10 repeats; the range is over all 50 folds.

| Model | AUC | AUC range (50 folds) | KS | KS range | Brier |
|---|---|---|---|---|---|
| Logistic regression | 0.780 ± 0.007 | 0.697 to 0.844 | 0.469 ± 0.015 | 0.317 to 0.600 | 0.169 |
| **XGBoost (constrained)** | **0.789 ± 0.008** | 0.723 to 0.859 | **0.485 ± 0.017** | 0.379 to 0.619 | 0.164 |
| XGBoost (unconstrained, reference) | 0.790 ± 0.010 | 0.717 to 0.866 | 0.481 ± 0.018 | 0.391 to 0.638 | 0.164 |

- XGBoost beats LR by 0.009 AUC on average (SD 0.018 across folds), winning in 68% of folds. The difference is small next to the fold-to-fold spread.
- The constraints cost 0.0009 AUC (SD 0.005). On the full data, the constrained model makes 0 wrong-direction moves on duration and installment rate, against 344 and 49 without constraints.
- Single folds range from 0.72 to 0.86 AUC because each test fold has only 200 rows. Use the averages, not single folds.

## 4. File for 채민규: `reports/german/oof_predictions.csv`

| Column | Meaning |
|---|---|
| `row_id` | Row number in `german.data` (0 to 999) |
| `pd_xgb`, `pd_lr` | Out-of-fold probability of default, averaged over the 10 repeats. Every value comes from models that never saw that row. |
| `default` | Actual outcome (1 = default) |
| `sex` | female / male (from attribute 9; not a model feature) |
| `age` | Age in years (a model feature) |
| `personal_status` | Full attribute 9 label (sex and marital status) |
| `foreign_worker` | yes / no (not a model feature) |

- The averaged PD is well calibrated: mean 29.5% (XGBoost) and 30.0% (LR) against an actual 30%. Its AUC is 0.795 (XGBoost) and 0.783 (LR), slightly higher than single runs because averaging reduces noise.
- **No approval cutoff and no decision column.** The threshold is for 채민규 to choose.

**UCI cost matrix, relevant for the threshold:** UCI states that approving a bad borrower is **5 times worse** than rejecting a good one (cost 5 vs. 1). With calibrated PD, approving costs 5 × PD and rejecting costs 1 × (1 − PD), so the break-even is **PD = 1/6 ≈ 0.167**, not 0.5. A cost-aware threshold therefore rejects many more applicants than a 0.5 cutoff, which changes approval rates per group and so the fairness picture. This is information only; no cutoff was applied.

## 5. Limitations

- **Only 1,000 rows:** single folds vary widely (AUC 0.72 to 0.86), and the smaller groups are small (125 applicants aged 50+, 37 non-foreign workers, 50 divorced or separated men).
- **No single women in the data:** "female" means divorced, separated or married women only, while "male" includes single men. Sex is therefore partly mixed up with marital status, and a gender gap may partly be a marital-status gap.
- **Proxies remain:** sex and foreign worker are not features, but other features (for example housing, job or age) can partly carry the same information.
- **Old and local data:** Germany, 1994, amounts in Deutsche Mark. Results do not describe today's lending.
- **Separate from GMSC:** different features and target definition, so the numbers are not comparable with the GMSC model.
