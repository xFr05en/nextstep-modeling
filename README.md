# NextStep: 데이터 · 시뮬레이터 · 모델링 (김원빈 파트)

> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다. 대체 데이터 변수와 성별은 실제 값이 아닌 시뮬레이션 값입니다.

4무원 팀의 씬파일러 대상 XAI 기반 대안신용평가 시스템(캡스톤디자인II / 산학프로젝트, CSE4187, 서강대학교) 중 데이터 정제, 대체 데이터 시뮬레이터, 변수별 조치 가능성 메타데이터, 단조 제약 모델 학습, PD에서 신용점수로의 변환을 담당합니다.

## 실행 방법

1. `data/raw/`에 `cs-training.csv`(Kaggle "Give Me Some Credit")를 넣습니다. 보조 German Credit 모델용 `german.data`도 같은 위치에 넣습니다 ([UCI 저장소](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)에서 내려받음).
2. 설정: macOS/Linux `bash setup.sh`, Windows `setup.bat`. 이후 `source .venv/bin/activate`.
3. 파이프라인을 순서대로 실행합니다 (전체 약 4분, 시드 고정):

```bash
python -m src.data
```
```bash
python -m src.data.simulator
```
```bash
python -m src.train --stage all
```

4. 선택: 보고서와 그림

```bash
python notebooks/eda.py
```
```bash
python notebooks/simulator_report.py
```
```bash
python notebooks/model_report.py
```
```bash
python -m src.actionability_table
```

5. 선택: German Credit 보조 모델 (GMSC와 별개, 약 20초)

```bash
python -m src.german
```

## 테스트

프로젝트 루트에서 실행합니다 (약 10초):

```bash
pytest
```

일부 테스트는 커밋되지 않는 파일(`data/processed/gmsc_clean.csv`, `data/processed/gmsc_sim.csv`)이 필요합니다. 위 파이프라인 명령으로 만듭니다.

- **기본값:** 해당 파일이 없으면 그 테스트는 실행할 명령을 알려주는 메시지와 함께 **건너뜁니다(skip)**.
- **`REQUIRE_DATA=1`:** 파일이 없으면 해당 테스트가 건너뛰지 않고 **실패**합니다. 데이터 누락이 조용히 통과되지 않도록 Docker와 CI에서 설정하십시오.

```bash
REQUIRE_DATA=1 pytest
```

Dockerfile에서는 `ENV REQUIRE_DATA=1`, GitHub Actions에서는 테스트 단계에 `env: REQUIRE_DATA: "1"`을 넣습니다.

## 결과 (최종 모델: XGBoost, 변수 19개 전부, 불균형 처리 없음, 단조 제약 11개)

| 헌장 목표 | 결과 |
|---|---|
| AUC ≥ 0.78 | 0.927 ± 0.004 |
| KS ≥ 0.28 | 0.704 |
| 씬파일러 AUC 향상 ≥ +0.03 | +0.071 |
| PSI < 0.1 | 0.0005 |
| 단조 제약으로 인한 AUC 손실 ≤ 0.01 | 0.0001 |
| pytest: 테스트 10개 이상, 80% 통과 | 테스트 36개, 100% 통과 |

대체 데이터가 실제 결과로부터 시뮬레이션되었으므로 이 수치는 낙관적입니다. 현실적인 기준은 GMSC만 사용한 AUC 0.865입니다. 아래 한계를 참고하십시오.

## 설정 파일 (모든 설정은 코드가 아니라 여기에 있음)

| 파일 | 내용 |
|---|---|
| `config/data.yaml` | 정제 규칙 (특수코드, 사용률 기준, 최소 연령) |
| `config/simulator.yaml` | 대체 변수: 분포, 잠재 상관 목표, 신용 연결, 씬파일러 규칙 |
| `config/actionability.yaml` | 변수별 조치 가능성 분류, 방향, 단위, 허용 범위, 단조 부호, 비고 (한국어, 영어) |
| `config/train.yaml` | 모델 비교 조합, 하이퍼파라미터, 폴드, 최종 모델 선택 규칙, MLflow 이름 |
| `config/scoring.yaml` | PD에서 점수(0~1000), 등급 A~E, 승인 규칙 |

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| `config/` | 위의 YAML 파일 5개 |
| `data/` | `raw/`, `processed/` (커밋하지 않음) |
| `src/` | `data.py`, `simulator.py`, `features.py`, `train.py`, `evaluate.py`, `scoring.py`, `actionability_table.py` |
| `notebooks/` | 보고서·그림 스크립트 (`src/`는 이것에 의존하지 않음) |
| `tests/` | pytest 테스트 |
| `models/` | `xgboost_v1.0.joblib` (버전 포함: `{알고리즘}_v{model_version}.joblib`, 버전은 `config/train.yaml`) |
| `reports/` | 단계별 보고서 (한국어, 영어), 그림, 결과표, 조치 가능성 Excel |
| `mlruns/` | MLflow 기록 (SQLite, 커밋하지 않음, `src.train`으로 다시 생성) |

## 보고서

| 단계 | 보고서 |
|---|---|
| 1. 데이터 정제 | `reports/data_report_ko.md` |
| 2. 탐색적 분석 | `reports/eda_report_ko.md` |
| 3. 시뮬레이터와 씬파일러 플래그 | `reports/simulator_report_ko.md` |
| 5. 모델 비교, 제약, 점수 체계 | `reports/model_report_ko.md` |
| MLflow 안내 | `reports/mlflow_schema_ko.md` |
| 조치 가능성 표 | `reports/actionability_table_ko.xlsx` (YAML에서 생성, 직접 수정 금지) |
| 점검 (경로, 공정성, 안정성, 홀드아웃, 민감도) | `reports/audit_report_ko.md` |
| German Credit 보조 모델 | `reports/german_report_ko.md` |

모든 보고서는 영어 버전(`_en`)도 있습니다.

## 팀원 연계

**채민규 (공정성, SHAP, DiCE, 비용 함수)**
- `config/actionability.yaml`: DiCE가 바꿀 수 있는 변수(`dice_vary`), 방향, 단위, 허용 범위, 난이도, 단계당 개월. 같은 내용이 Excel에도 있습니다.
- 경로 목표: `config/scoring.yaml`의 `recourse_target_score` = 495. 승인 자체는 등급 A~C (점수 475 이상)로 유지하며, 20점 여유는 모델을 재학습해도 경로가 승인되도록 하기 위함입니다 (`reports/audit_report_ko.md` 참고). PD를 점수, 등급, 승인으로 바꿀 때 `src/scoring.py`를 사용합니다.
- 자동이체는 모델 변수지만 경로에는 쓰지 않습니다: 분류 권장 불가(NOT_RECOMMENDED), `dice_vary: false`, 단조 -1. SHAP 거절 사유에서는 자동이체를 조치 가능 항목이 아니라 "참고(변경 불가)"에 표시하여, 설명이 자동이체 설정을 권하지 않도록 하십시오.
- 모델: `models/xgboost_v1.0.joblib`, 입력 열 21개를 DataFrame으로 받는 파이프라인. `predict_proba[:, 1]`이 PD입니다.
- 공정성 입력: `gender_female`(시뮬레이션, 보호 속성, 모델 변수 아님), `age`(보호 속성, 모델 변수로 사용). 연령은 사용률, 부양가족 수와 관련됩니다 (EDA와 시뮬레이터 보고서 참고).
- 실제 성별 점검: `reports/german/oof_predictions.csv`에 German Credit 신청자 1,000명의 out-of-fold PD (XGBoost, LR), 실제 결과, 성별, 연령, 혼인 상태, 외국인 노동자 여부가 있습니다. 기준점은 적용하지 않았으며, UCI 비용 행렬 (손익분기 PD 약 0.167)은 `reports/german_report_ko.md`를 참고하십시오.

**윤제진 (MLflow, Docker, FastAPI, Streamlit)**
- `reports/mlflow_schema_ko.md`: 실험과 실행 이름, 지표 키, 입력 열, 모델 불러오는 방법. 서빙 모델은 확률을 반환합니다 (1번 열 = PD).
- MLflow 3.16이 일반 폴더 백엔드를 더 이상 받지 않아 `mlruns/mlflow.db`(SQLite)를 사용합니다. 보기: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- Docker와 CI에서 `REQUIRE_DATA=1`을 설정하십시오 (테스트 절 참고).

## 한계

- **시뮬레이션 편향.** 대체 변수는 실제 부도 열로부터 생성되므로, 이를 쓰는 모델은 실제 데이터보다 좋아 보입니다. 절대 수치보다 상충 관계의 패턴(예: 신용 연결이 향상 폭을 줄이는 정도)이 더 신뢰할 만합니다.
- **상관 규칙은 코퓰라의 잠재 척도에 적용합니다** (5개 변수 모두 0.35). 부도와의 관측 Pearson 상관은 약 0.18~0.27입니다.
- **PSI는 무작위 폴드를 사용**하므로 구조적으로 0에 가깝습니다. GMSC에는 날짜가 없어 시간에 따른 변화는 검증할 수 없습니다.
- **정책적 선택은 가정입니다:** 등급 기준, 승인 규칙, 단계당 기간, 난이도는 팀의 가정이며 금융기관의 값이 아닙니다.
- German Credit은 별도의 작은 모델이며 (1,000행, 미혼 여성 없음) GMSC와 수치를 비교할 수 없습니다. `reports/german_report_ko.md` 참고.

## 데이터 출처

German Credit: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0). GMSC: Kaggle "Give Me Some Credit".
