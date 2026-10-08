# MLflow Guide (for the serving and registry work)

> Educational use only. Not for real financial decisions.

## Where the runs are

- **Tracking location:** the environment variable `MLFLOW_TRACKING_URI`.
  - **Unset, or a folder path** (as in `.env.example`, `./mlruns`): a local SQLite database `mlruns/mlflow.db`, with artifacts in `mlruns/artifacts/`, relative to the project root. MLflow 3.16 refuses the plain folder backend, which is why the folder holds a database.
  - **A URI** (for example `http://mlflow:5000`): used as is; the server decides where artifacts go.
- **Inside docker-compose:** set `MLFLOW_TRACKING_URI` to the compose MLflow server and **rerun training** (`python -m src.train --stage all`, about 3 minutes, seeded). Do not copy the local `mlruns/` folder into a container: its database stores **absolute** artifact paths from the machine where it was created, so the artifacts would not be found.
- `mlruns/` is gitignored. Browse it locally with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## Experiments and runs

| Experiment | Runs | Run name pattern |
|---|---|---|
| `nextstep-lr` | 9 comparison runs | `lr__{features}__{resampling}` |
| `nextstep-lgbm` | 9 comparison runs + monotonic runs | `lgbm__...`, monotonic: `lgbm__both__{resampling}__mono-{set}` |
| `nextstep-xgb` | 9 comparison runs + monotonic runs | `xgb__...`, monotonic: `xgb__both__{resampling}__mono-{set}` |
| `nextstep-final` | 1 | `{model}__both__{resampling}__mono-{set}[__platt]__final`; currently `xgb__both__none__mono-full__final` |
| `nextstep-audit` | audit runs (Step 7) | |

Values: features = `gmsc`, `alt`, `both`; resampling = `none`, `class_weight`, `smote`. Comparison runs use 5-fold CV inside the **train** split; monotonic runs are evaluated on **validation**; the final run is trained on train+validation and evaluated once on **test** (split defined by `data/processed/split.csv`).

## Tags, params, metrics

- Tags: `step`, `stage` (compare / monotonic / final), `owner=wonbin`; the best run per algorithm has `best_of_algorithm=true`; the final run has `trained_on=train+validation`, `evaluated_on=test`.
- Params: `model`, `feature_set`, `resampling`, `monotone_set`, `seed`, `n_splits`, `sim_b`, `data_md5`, hyperparameters with prefix `hp_`; the final run adds `model_version`, `model_file`, `calibrated`.
- Comparison runs: per fold (`step` = fold) `auc`, `ks`, `thin_auc`, `psi`, `brier`, `precision`, `recall`, `f1`, and `*_mean` / `*_std` of each; `both` runs add `thin_auc_gain_mean/std`.
- Monotonic runs: `val_auc_unconstrained`, `val_auc_constrained`, `mono_auc_loss` with `mono_auc_loss_ci_low/high`.
- Final run: `test_auc`, `test_ks`, `test_thin_auc`, `test_thin_auc_lift`, `test_psi_trainval_vs_test`, `test_brier`, `test_precision`, `test_recall`, `test_f1`, `mono_auc_loss`, `approval_rate`, `approval_rate_thin`.
- Precision, recall and F1 treat **default as the positive class**, with "predicted default" = rejected by the fixed score-475 rule in `config/scoring.yaml`.

## Model artifacts

- **Per algorithm:** the best `both` run of each algorithm holds a model artifact, refit on the full train split.
- **Final:** the final run holds the shipped model (also saved as `models/xgboost_v1.0.joblib`, model version 1.0).
- Every artifact has a **signature** (21 input columns, output = 2 probability columns) and an **input_example** (5 training rows of the feature columns).
- pyfunc serves `predict_proba`, not class labels; column 1 is the PD.
- MLflow saves with skops; the model's own classes are listed as trusted at save time (imblearn.pipeline.Pipeline, numpy.dtype, xgboost.core.Booster, xgboost.sklearn.XGBClassifier).

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")      # or the compose server
model = mlflow.pyfunc.load_model("runs:/ab9fa56de2ee43f0823ce56096a9457b/model")
example = model.input_example                                 # DataFrame with the 21 columns
pd_ = model.predict(example)[:, 1]                            # probability of default
```

The run id changes on every retraining; find the final run by experiment `nextstep-final` and tag `stage=final`. Turn PD into score, grade and approval with `src/scoring.py` and `config/scoring.yaml`. The Model Registry (Production / Staging) is 윤제진's part.

## Input columns (21, this order, all numeric; NaN allowed)

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

`autopay_ratio` and `gender_female` are in the dataset but are not model inputs.
