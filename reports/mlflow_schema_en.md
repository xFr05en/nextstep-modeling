# MLflow Guide (for the serving and registry work)

> Educational use only. Not for real financial decisions.

## Where the runs are

- Backend: SQLite at `mlruns/mlflow.db`, artifacts in `mlruns/artifacts/`. MLflow 3.16 no longer accepts the plain folder backend, and the model registry needs a database backend anyway. `mlruns/` is gitignored, so recreate it with `python -m src.train --stage all` (about 3 minutes, same results because everything is seeded).
- Browse: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`
- In code: `mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")`

## Experiments and run names

| Experiment | Runs | Run name pattern |
|---|---|---|
| `nextstep-model-comparison` | 27 | `{model}__{features}__{resampling}`, e.g. `xgb__both__none` |
| `nextstep-monotonic` | 1 unconstrained + 1 or more constrained per boosting model | `{model}__both__{resampling}__mono-{set}`, set = `none`, `full`, or `no-<dropped features>` |
| `nextstep-final` | 1 | `{model}__both__{resampling}__mono-{set}[__platt]__final`; currently `xgb__both__none__mono-full__final` |

Values: model = `lr`, `lgbm`, `xgb`. features = `gmsc`, `alt`, `both`. resampling = `none`, `class_weight`, `smote`.

## Tags and params

- Tags: `step=5`, `stage=compare|monotonic|final`, `owner=wonbin`
- Params: `model`, `feature_set`, `resampling`, `monotone_set`, `seed`, `n_splits`, `sim_b` (simulator credit link), `data_md5` (first 8 characters of the training data hash), and every hyperparameter with prefix `hp_`. Monotonic runs add `dropped_constraints`. The final run adds `calibrated`.

## Metric keys (constants in `src/evaluate.py`)

| Key | Meaning |
|---|---|
| `auc`, `ks`, `thin_auc`, `psi`, `brier` | Per fold, logged with `step` = fold number (0 to 4) |
| `auc_mean`, `auc_std`, `ks_mean`, `ks_std`, `thin_auc_mean`, `thin_auc_std`, `psi_mean`, `psi_std`, `brier_mean`, `brier_std` | Mean and SD over the 5 folds |
| `thin_auc_gain_mean`, `thin_auc_gain_std` | Thin-filer AUC of `both` minus `gmsc`, paired by fold (only on `both` runs and the final run) |
| `mono_auc_loss` | Unconstrained AUC minus constrained AUC (monotonic and final runs) |
| `approval_rate`, `approval_rate_thin` | Final run only, using `config/scoring.yaml` |

## Artifacts

- Every run: `per_fold_metrics.csv`, `features.json`. Constrained runs: `monotone.json`.
- Final run: `model/` (MLflow sklearn model), `final_summary.json`, `grade_table.csv`, `scoring.yaml`, `actionability.yaml`.

## Using the final model

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
model = mlflow.pyfunc.load_model("runs:/0540be4848544f999d99bf1ea413008d/model")
proba = model.predict(X)      # shape (n, 2); column 1 = probability of default
pd_ = proba[:, 1]
```

- The pyfunc serves `predict_proba` (set with `pyfunc_predict_fn`), not class labels. Column 1 is the PD.
- To turn PD into score, grade and approval, use `src/scoring.py` with `config/scoring.yaml` (`score_frame(pd_)`).
- MLflow saves the model with skops. Its own classes are listed as trusted at save time (`imblearn.pipeline.Pipeline, numpy.dtype, xgboost.core.Booster, xgboost.sklearn.XGBClassifier`).
- The same fitted model is also saved as `models/final_model.joblib`.
- The run ID changes every time training is rerun. Find the final run by experiment `nextstep-final` and tag `stage=final`, not by a fixed ID.

## Input columns (21, this order, all numeric; missing values allowed as NaN)

1. `RevolvingUtilizationOfUnsecuredLines`
2. `age`
3. `NumberOfTime30-59DaysPastDueNotWorse`
4. `DebtRatio`
5. `MonthlyIncome`
6. `NumberOfOpenCreditLinesAndLoans`
7. `NumberOfTimes90DaysLate`
8. `NumberRealEstateLoansOrLines`
9. `NumberOfTime60-89DaysPastDueNotWorse`
10. `NumberOfDependents`
11. `pastdue_special_code`
12. `income_missing`
13. `income_zero`
14. `util_outlier`
15. `telecom_payment_rate`
16. `utility_payment_rate`
17. `telecom_tenure_months`
18. `insurance_paid_months`
19. `spending_consistency`
20. `regular_payment_count`
21. `app_login_frequency`

The pipeline does its own median imputation, so NaN is fine. Gender is not an input.
