# GMSC Data Report (Step 1: Loading and Cleaning)

> Educational use only. Not for real financial decisions.

- Source: Kaggle "Give Me Some Credit", `cs-training.csv`
- Code: `src/data.py` (run `python -m src.data`), rules in `config/data.yaml`
- Output: `data/processed/gmsc_clean.csv`, machine-readable summary in `reports/data_summary.json`

## 1. Overview

| Item | Value |
|---|---|
| Rows (raw) | 150,000 |
| Rows (clean) | 149,999 |
| Columns (clean) | 11 original + 4 flags = 15 |
| Target | `SeriousDlqin2yrs` (1 = serious delinquency within 2 years) |
| Default rate | 6.68% (strong class imbalance) |

## 2. Issues found and rules applied

Every issue was checked against the actual file before any rule was written.

| # | Issue | Rows | Rule | New flag |
|---|---|---|---|---|
| 1 | Values 96 / 98 in the three past-due columns are special codes, not counts. In all 269 rows the three columns carry the code together. | 269 | Set the three columns to missing | `pastdue_special_code` |
| 2 | `MonthlyIncome` missing | 29,731 (19.82%) | Leave missing (filled later on training data only) | `income_missing` |
| 3 | `MonthlyIncome` = 0 | 1,634 | Keep 0 as a real value | `income_zero` |
| 4 | `DebtRatio` holds a raw amount, not a ratio, when income is missing or 0 (median 1,159 vs 0.30) | 31,365 | Set `DebtRatio` to missing | (covered by flags 2 and 3) |
| 5 | `RevolvingUtilizationOfUnsecuredLines` above 10 (max 50,708) | 241 | Set to missing | `util_outlier` |
| 6 | Utilization between 1 and 10 (over the credit limit) | 3,080 | Keep, realistic | none |
| 7 | age = 0 | 1 | Drop the row | none |
| 8 | Fully duplicated rows | 609 | Keep, only counted | none |
| 9 | `Unnamed: 0` row id column | all | Drop the column | none |
| 10 | `NumberOfDependents` missing | 3,924 (2.62%) | Leave missing (filled later) | none |

Why missing values are not filled here: a fill value such as the median, computed on the whole dataset, would carry information from the test data into training. Filling will be fit on training folds only, in `src/features.py`.

Note on duplicates: after cleaning, 1,925 rows look identical. The extra rows are not new duplicates. They differed only in a value that was set to missing (for example a raw-amount `DebtRatio`).

## 3. Default rate by flag

| Flag | Rate when 0 | Rate when 1 |
|---|---|---|
| `pastdue_special_code` | 6.60% | **54.65%** |
| `income_missing` | 6.95% | 5.61% |
| `income_zero` | 6.71% | 4.04% |
| `util_outlier` | 6.68% | 7.05% |

The 96/98 code group defaults at about 8 times the base rate, so the flag carries strong signal and should stay in the model.

## 4. Missing values after cleaning

| Column | Missing |
|---|---|
| `DebtRatio` | 20.91% |
| `MonthlyIncome` | 19.82% |
| `NumberOfDependents` | 2.62% |
| Three past-due columns | 0.18% each |
| `RevolvingUtilizationOfUnsecuredLines` | 0.16% |

## 5. Changes to `config/actionability.yaml`

- Utilization `bounds` changed from [0, 1] to [0, 10] to match the cleaning threshold. `direction: decrease` still prevents DiCE from suggesting an increase.
- The four flag columns were added as `IMMUTABLE`, `source: derived`, `dice_vary: false`, `monotone: 0`, so DiCE never changes them.

## 6. Open items

- **Extreme `DebtRatio` with income present:** 659 rows above 10 and 468 above 100 (99.9th percentile 1,467). This probably comes from a very small reported income. They are left as they are for now. Tree models are not affected much, but Logistic Regression may need a cap or log transform in Step 5.
- **Thin-filer definition:** see section 7.

## 7. Thin-filer definition

The mission defines a thin-filer as a customer with **2 or more missing past-due columns OR less than 12 months of card history**. This rule is implemented as `is_thin_filer_mission()` in `src/features.py`, but it cannot identify thin-filers in GMSC:

| Data | Mission rule flags | Notes |
|---|---|---|
| Raw `cs-training.csv` | **0 rows (0%)** | The past-due columns have no missing values, and there is no card-history column. |
| After cleaning (Step 1) | **269 rows (0.18%)** | Exactly the rows with code 96/98, set to missing in Step 1. They default at **54.65%**, the riskiest group in the data, not people with short credit histories. |
| Simulator, `thin_filer_mode="mask"`, `thin_filer_ratio=0.3` | **30.0%** (44,731 rows masked) | Demonstration only: the past-due columns are blanked for a random, seeded share of rows. Because the choice is random, these rows default at 6.89%, about the average. |

**Therefore every official number uses our proxy:** at most 2 open loans or credit lines **and** no real estate loans (`is_thin_filer_proxy()`, rule in `config/simulator.yaml`). It describes people with very few credit products, which is what "thin file" means.
- It flags **11,564 rows (7.71%)** with a default rate of **12.93%**, about twice the 6.68% average.
- It also flags all **269** special-code rows, so both rules agree on them.
- The alternative-data lift (thin-filer AUC with vs. without alternative data) is measured on the proxy group.

**Simulator modes:** `thin_filer_ratio` works in two ways (`thin_filer_mode` in `config/simulator.yaml`).
- **`subsample` (default):** reaches the ratio by subsampling without replacement, using the proxy. The final model and every official number use this mode with the natural share.
- **`mask` (demonstration only):** blanks the 3 past-due columns after the alternative variables are generated, so the mission rule flags the requested share. It shows how the mission's own rule would behave if bureau data were missing. It is never used for the final model.
