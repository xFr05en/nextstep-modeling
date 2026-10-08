# MLflow 안내 (서빙과 레지스트리 작업용)

> 교육용으로만 사용합니다. 실제 금융 의사결정에 사용하지 마십시오.

## 실행 기록 위치

- **기록 위치:** 환경 변수 `MLFLOW_TRACKING_URI`.
  - **설정하지 않거나 폴더 경로인 경우** (`.env.example`의 `./mlruns`처럼): 프로젝트 루트 기준 `mlruns/mlflow.db` (로컬 SQLite), 아티팩트는 `mlruns/artifacts/`. MLflow 3.16이 일반 폴더 백엔드를 받지 않아 폴더 안에 데이터베이스를 둡니다.
  - **URI인 경우** (예: `http://mlflow:5000`): 그대로 사용하며, 아티팩트 위치는 서버가 정합니다.
- **docker-compose 안에서는:** `MLFLOW_TRACKING_URI`를 compose의 MLflow 서버로 설정하고 **학습을 다시 실행**하십시오 (`python -m src.train --stage all`, 약 3분, 시드 고정). 로컬 `mlruns/` 폴더를 컨테이너로 복사하지 마십시오. 그 데이터베이스는 만들어진 컴퓨터의 **절대** 아티팩트 경로를 저장하므로 아티팩트를 찾을 수 없습니다.
- `mlruns/`는 git에서 제외됩니다. 로컬에서 보기: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## 실험과 실행

| 실험 | 실행 | 실행 이름 형식 |
|---|---|---|
| `nextstep-lr` | 비교 실행 9개 | `lr__{features}__{resampling}` |
| `nextstep-lgbm` | 비교 실행 9개 + 단조 제약 실행 | `lgbm__...`, 단조: `lgbm__both__{resampling}__mono-{set}` |
| `nextstep-xgb` | 비교 실행 9개 + 단조 제약 실행 | `xgb__...`, 단조: `xgb__both__{resampling}__mono-{set}` |
| `nextstep-final` | 1 | `{model}__both__{resampling}__mono-{set}[__platt]__final`, 현재 `xgb__both__none__mono-full__final` |
| `nextstep-audit` | 점검 실행 (7단계) | |

값: features = `gmsc`, `alt`, `both`; resampling = `none`, `class_weight`, `smote`. 비교 실행은 **train** 분할 안의 5겹 교차검증, 단조 제약 실행은 **validation**에서 평가, 최종 실행은 train+validation으로 학습하고 **test**에서 한 번만 평가합니다 (분할은 `data/processed/split.csv`).

## 태그, 파라미터, 지표

- 태그: `step`, `stage` (compare / monotonic / final), `owner=wonbin`. 알고리즘별 최고 실행은 `best_of_algorithm=true`, 최종 실행은 `trained_on=train+validation`, `evaluated_on=test`.
- 파라미터: `model`, `feature_set`, `resampling`, `monotone_set`, `seed`, `n_splits`, `sim_b`, `data_md5`, `hp_` 접두사의 하이퍼파라미터. 최종 실행에는 `model_version`, `model_file`, `calibrated` 추가.
- 비교 실행: 폴드별 (`step` = 폴드) `auc`, `ks`, `thin_auc`, `psi`, `brier`, `precision`, `recall`, `f1`과 각각의 `*_mean` / `*_std`. `both` 실행에는 `thin_auc_gain_mean/std` 추가.
- 단조 제약 실행: `val_auc_unconstrained`, `val_auc_constrained`, `mono_auc_loss`와 `mono_auc_loss_ci_low/high`.
- 최종 실행: `test_auc`, `test_ks`, `test_thin_auc`, `test_thin_auc_lift`, `test_psi_trainval_vs_test`, `test_brier`, `test_precision`, `test_recall`, `test_f1`, `mono_auc_loss`, `approval_rate`, `approval_rate_thin`.
- 정밀도, 재현율, F1은 **부도를 양성 클래스**로 보며, "부도 예측" = `config/scoring.yaml`의 고정 규칙(점수 475 미만)으로 거절된 경우입니다.

## 모델 아티팩트

- **알고리즘별:** 각 알고리즘의 최고 `both` 실행에 train 분할 전체로 다시 학습한 모델 아티팩트가 있습니다.
- **최종:** 최종 실행에 배포 모델이 있습니다 (`models/xgboost_v1.0.joblib`로도 저장, 모델 버전 1.0).
- 모든 아티팩트에 **signature** (입력 21열, 출력 = 확률 2열)와 **input_example** (변수 열의 학습 데이터 5행)이 있습니다.
- pyfunc는 클래스 라벨이 아니라 `predict_proba`를 반환하며, 1번 열이 PD입니다.
- MLflow는 skops로 저장하며, 모델 자신의 클래스를 저장 시 신뢰 목록에 넣었습니다 (imblearn.pipeline.Pipeline, numpy.dtype, xgboost.core.Booster, xgboost.sklearn.XGBClassifier).

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")      # 또는 compose 서버
model = mlflow.pyfunc.load_model("runs:/ab9fa56de2ee43f0823ce56096a9457b/model")
example = model.input_example                                 # 21열 DataFrame
pd_ = model.predict(example)[:, 1]                            # 부도 확률
```

실행 ID는 다시 학습할 때마다 바뀌므로 실험 `nextstep-final`과 태그 `stage=final`로 최종 실행을 찾으십시오. PD를 점수, 등급, 승인으로 바꿀 때는 `src/scoring.py`와 `config/scoring.yaml`을 사용합니다. 모델 레지스트리 (Production / Staging)는 윤제진 담당입니다.

## 입력 열 (21개, 이 순서, 모두 숫자, NaN 허용)

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

`autopay_ratio`와 `gender_female`은 데이터셋에 있지만 모델 입력이 아닙니다.
