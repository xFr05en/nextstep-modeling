# MLflow 안내 (서빙과 레지스트리 작업용)

> 교육용으로만 사용합니다. 실제 금융 의사결정에 사용하지 마십시오.

## 실행 기록 위치

- 백엔드: `mlruns/mlflow.db` (SQLite), 아티팩트는 `mlruns/artifacts/`. MLflow 3.16은 일반 폴더 백엔드를 더 이상 받지 않으며, 모델 레지스트리도 데이터베이스 백엔드가 필요합니다. `mlruns/`는 git에서 제외되므로 `python -m src.train --stage all`로 다시 만듭니다 (약 3분, 시드가 고정되어 결과 동일).
- 보기: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`
- 코드에서: `mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")`

## 실험과 실행 이름

| 실험 | 실행 수 | 실행 이름 형식 |
|---|---|---|
| `nextstep-model-comparison` | 27 | `{model}__{features}__{resampling}`, 예: `xgb__both__none` |
| `nextstep-monotonic` | 부스팅 모델별 제약 없음 1개 + 제약 1개 이상 | `{model}__both__{resampling}__mono-{set}`, set = `none`, `full`, `no-<제거한 변수>` |
| `nextstep-final` | 1 | `{model}__both__{resampling}__mono-{set}[__platt]__final`, 현재 `xgb__both__none__mono-full__final` |

값: model = `lr`, `lgbm`, `xgb`. features = `gmsc`, `alt`, `both`. resampling = `none`, `class_weight`, `smote`.

## 태그와 파라미터

- 태그: `step=5`, `stage=compare|monotonic|final`, `owner=wonbin`
- 파라미터: `model`, `feature_set`, `resampling`, `monotone_set`, `seed`, `n_splits`, `sim_b` (시뮬레이터 신용 연결 강도), `data_md5` (학습 데이터 해시 앞 8자리), 그리고 `hp_` 접두사가 붙은 모든 하이퍼파라미터. 단조 제약 실행에는 `dropped_constraints`, 최종 실행에는 `calibrated`가 추가됩니다.

## 지표 키 (상수는 `src/evaluate.py`)

| 키 | 의미 |
|---|---|
| `auc`, `ks`, `thin_auc`, `psi`, `brier` | 폴드별 값, `step` = 폴드 번호 (0~4) |
| `auc_mean`, `auc_std`, `ks_mean`, `ks_std`, `thin_auc_mean`, `thin_auc_std`, `psi_mean`, `psi_std`, `brier_mean`, `brier_std` | 5개 폴드의 평균과 표준편차 |
| `thin_auc_gain_mean`, `thin_auc_gain_std` | `both`의 씬파일러 AUC에서 `gmsc`를 뺀 값, 폴드별 짝지음 (`both` 실행과 최종 실행에만) |
| `mono_auc_loss` | 제약 없는 AUC에서 제약 AUC를 뺀 값 (단조 제약 실행과 최종 실행) |
| `approval_rate`, `approval_rate_thin` | 최종 실행에만, `config/scoring.yaml` 기준 |

## 아티팩트

- 모든 실행: `per_fold_metrics.csv`, `features.json`. 제약 실행: `monotone.json`.
- 최종 실행: `model/` (MLflow sklearn 모델), `final_summary.json`, `grade_table.csv`, `scoring.yaml`, `actionability.yaml`.

## 최종 모델 사용법

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
model = mlflow.pyfunc.load_model("runs:/e137818745b6445d83d2824a9fe97428/model")
proba = model.predict(X)      # 크기 (n, 2); 1번 열 = 부도 확률
pd_ = proba[:, 1]
```

- pyfunc는 클래스 라벨이 아니라 `predict_proba`를 반환합니다 (`pyfunc_predict_fn`으로 설정). 1번 열이 PD입니다.
- PD를 점수, 등급, 승인으로 바꾸려면 `src/scoring.py`와 `config/scoring.yaml`을 사용합니다 (`score_frame(pd_)`).
- MLflow는 모델을 skops로 저장합니다. 모델 자신의 클래스를 저장 시 신뢰 목록에 넣었습니다 (`imblearn.pipeline.Pipeline, numpy.dtype, xgboost.core.Booster, xgboost.sklearn.XGBClassifier`).
- 같은 모델이 `models/final_model.joblib`에도 저장되어 있습니다.
- 학습을 다시 실행하면 실행 ID가 바뀝니다. 고정 ID 대신 실험 `nextstep-final`과 태그 `stage=final`로 최종 실행을 찾으십시오.

## 입력 열 (19개, 이 순서, 모두 숫자; 결측은 NaN 허용)

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
15. `telecom_ontime_rate`
16. `utility_ontime_rate`
17. `telecom_tenure_months`
18. `insurance_paid_months`
19. `autopay_ratio`

파이프라인이 자체적으로 중앙값 대체를 하므로 NaN이 있어도 됩니다. 성별은 입력이 아닙니다.
