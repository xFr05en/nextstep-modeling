# ANOVA 인계 문서 (체민규용 데이터)

> 교육용으로만 사용합니다. 실제 금융 의사결정에 사용하지 마십시오. 김원빈은 데이터만 제공하며, 모든 ANOVA는 체민규가 수행합니다.

**파일:** `reports/cv_fold_auc.csv` (`python -m src.train --stage compare`로 생성). 실행 × 반복 × 폴드 × 세그먼트당 1행: 27개 실행 × 5회 반복 × 5폴드 × 3세그먼트 = 2,025행 (`config/train.yaml`의 `cv_repeats: 5`).

| 열 | 의미 |
|---|---|
| `run` | `{model}__{features}__{resampling}`, MLflow 실행 이름과 같음 |
| `model` | `lr`, `xgb`, `lgbm` |
| `features` | `gmsc` (전통 Only), `alt` (대안 Only), `both` (통합) |
| `resampling` | `none`, `class_weight`, `smote` |
| `repeat`, `fold` | 교차검증 반복 0~4와 폴드 0~4 (실행당 25개 반복값) |
| `segment` | `all`, `thin` (대리 기준 씬파일러), `general` (씬파일러 아님) |
| `n_rows`, `n_defaults` | 해당 폴드·세그먼트의 검증 행 수와 부도 수 |
| `auc` | 해당 폴드의 검증 행 중 세그먼트에 속한 행의 AUC |

모든 폴드는 **train** 분할 (`data/processed/split.csv`) 안의 5회 반복 층화 5겹 교차검증 (타깃 × 씬파일러)이며, 모든 실행이 같은 폴드를 쓰므로 `repeat`과 `fold`로 짝지어집니다.

**검증별 사용 행:**

| 검증 | 행 | 집단 |
|---|---|---|
| 검증 1 (필수): 피처 그룹별 일원 ANOVA | `segment == "all"` | `features`: gmsc / alt / both |
| 검증 2 (필수): 알고리즘별 일원 ANOVA | `segment == "all"` | `model`: lr / xgb / lgbm |
| 보너스: 이원 ANOVA | `segment`이 thin, general이고 `features`가 gmsc, both | 세그먼트 × 피처 (최종 설정 `model == "xgb"`, `resampling == "none"` 권장) |

**미션 보고 규칙:** 각 검증에 F, p-value, η²를 보고합니다. p < 0.05이면 Tukey HSD를 수행합니다. 최소 1개 검증이 p < 0.05여야 합니다 (검증 1은 큰 차이로 충족). 리포트에 1페이지 이상의 해석을 포함합니다.

**검정력 점검 (보너스 이원, 최종 설정 xgb/none, 셀당 25):** 폴드별 AUC 표준편차 0.0072~0.0102 (씬파일러), 0.0031~0.0054 (일반). 씬파일러 향상 +0.0901, 일반 향상 +0.0817이므로 교호작용은 +0.0084. α = 0.05에서 근사 검정력: 5회 반복으로 **0.85** (3회 반복 0.61, 5겹 1회 0.24~0.38). 검증 1과 2는 효과가 매우 커서 어느 경우든 검정력이 충분합니다.

**한계 (리포트에 명시):**
1. **폴드 AUC는 독립 표본이 아닙니다.** 폴드들이 학습 데이터 대부분을 공유하므로 ANOVA p-value는 근사값입니다 (보통 실제보다 작음).
2. **이원 검증에서 한 폴드의 씬파일러 AUC와 일반 AUC는 같은 모델에서 나오므로** 역시 독립이 아닙니다. 선택 점검: 폴드별 교호작용 대비에 대한 짝지은 검정, 또는 폴드를 임의효과로 둔 혼합모형.

**팀 분할 규칙:** `data/processed/split.csv` (row_id, split)가 모두의 train / validation / test를 정합니다. 공정성 완화와 기준점은 **validation**에서 조정하고 **test**에서 보고합니다. 점수 475 승인 기준은 `config/scoring.yaml`의 고정 규칙입니다.
