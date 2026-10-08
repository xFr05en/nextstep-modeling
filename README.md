# NextStep: 데이터 · 시뮬레이터 · 모델링 (김원빈 파트)

> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다. 대체 데이터 변수와 성별은 실제 값이 아닌 시뮬레이션 값입니다.

4무원 팀의 씬파일러 대상 XAI 기반 대안신용평가 시스템(캡스톤디자인II / 산학프로젝트, CSE4187, 서강대학교) 중 데이터 로드와 정제, 대체 데이터 시뮬레이터, 변수별 조치 가능성 메타데이터, 단조 제약 모델 학습, PD에서 신용점수로의 변환을 담당합니다.

## 실행 방법

1. `data/raw/`에 `cs-training.csv`(Kaggle "Give Me Some Credit")를 넣습니다. 보조 German Credit 모델용 `german.data`도 같은 위치에 넣습니다 ([UCI 저장소](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)).
2. 설정: macOS/Linux `bash setup.sh`, Windows `setup.bat`. 이후 `source .venv/bin/activate`.
3. 파이프라인을 순서대로 실행합니다 (약 15분, 시드 고정):

```bash
python -m src.data
```
```bash
python -m src.data.simulator
```
```bash
python -m src.train --stage all
```

4. 선택: 보고서, 그림, 점검

```bash
python notebooks/eda.py
```
```bash
python notebooks/simulator_report.py
```
```bash
python notebooks/simulator_scenarios.py
```
```bash
python notebooks/model_report.py
```
```bash
python notebooks/audit.py
```
```bash
python -m src.actionability_table
```

5. 선택: German Credit 보조 모델 (GMSC와 별개, 약 20초)

```bash
python -m src.german
```

## 테스트

프로젝트 루트에서 실행합니다 (약 30초, 테스트 91개):

```bash
pytest
```

일부 테스트는 커밋되지 않는 파일(`data/processed/*.csv`, `mlruns/`)이 필요합니다. 위 파이프라인 명령으로 만듭니다.

- **기본값:** 해당 파일이 없으면 그 테스트는 실행할 명령을 알려주는 메시지와 함께 **건너뜁니다(skip)**.
- **`REQUIRE_DATA=1`:** 파일이 없으면 해당 테스트가 건너뛰지 않고 **실패**합니다. Docker와 CI에서 설정하십시오.

```bash
REQUIRE_DATA=1 pytest
```

Dockerfile에서는 `ENV REQUIRE_DATA=1`, GitHub Actions에서는 테스트 단계에 `env: REQUIRE_DATA: "1"`을 넣습니다.

## 결과 (공식: 테스트 세트, 배포 모델 `models/xgboost_v1.0.joblib`)

XGBoost, 변수 21개, 불균형 처리 없음, 단조 제약 13개. train + validation으로 학습하고 15% 테스트 세트에서 한 번만 평가. 95% 부트스트랩 구간.

| 헌장 목표 | 테스트 결과 |
|---|---|
| AUC ≥ 0.78 | 0.949 (0.944~0.953) |
| KS ≥ 0.28 | 0.760 (0.745~0.776) |
| 씬파일러 AUC 향상 ≥ +0.03 | +0.071 (+0.052~+0.092) |
| PSI < 0.1 | 0.0004 |
| 단조 제약으로 인한 AUC 손실 ≤ 0.01 | 0.0000 (validation) |
| pytest: 테스트 10개 이상, 80% 통과 | 테스트 91개, 100% 통과 |

**이 수치는 낙관적입니다.** 시뮬레이터가 테스트 세트를 포함한 모든 행에 실제 결과를 사용하며 (미션 예시와 같은 방식), 미션 변수 각각이 단독으로 AUC 0.80~0.84에 도달합니다. 현실적인 기준은 GMSC만 사용한 모델입니다 (CV AUC 0.864, 씬파일러 테스트 AUC 0.861). `reports/model_report_ko.md` 참고.

## 설정 파일 (모든 설정은 코드가 아니라 여기에 있음)

| 파일 | 내용 |
|---|---|
| `config/data.yaml` | 정제 규칙 (특수코드, 사용률 기준, 최소 연령) |
| `config/simulator.yaml` | 대체 변수 8개 (분포, 관측·잠재 상관 목표, 공통 요인), 신용 연결, `thin_filer_ratio` / `thin_filer_mode` / `bias_ratio`, 씬파일러 규칙 |
| `config/actionability.yaml` | 변수별 조치 가능성 분류, 방향, 단위, 허용 범위, 단조 부호, `dice_vary`, `model_feature`, 비고 (한국어, 영어) |
| `config/train.yaml` | 분할 (70/15/15), 교차검증 (5 × 5겹), 모델 비교 조합, 하이퍼파라미터, 선정 규칙, 모델 버전, MLflow 실험 |
| `config/scoring.yaml` | PD에서 점수(0~1000), 등급 A~E, 승인 규칙 (점수 475 이상), 경로 목표 (495) |
| `config/german.yaml` | German Credit 보조 모델 |
| `.env.example` | `MLFLOW_TRACKING_URI` (기본은 로컬 `./mlruns`, Docker에서는 compose의 MLflow 서버) |

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| `config/` | 위의 YAML 파일들 |
| `data/` | `raw/`, `processed/` (커밋하지 않음). `processed/split.csv`가 train / validation / test를 정함 |
| `src/data/` | `loader.py` (GMSC / German Credit 로드, 컬럼명·타입·결측 비율 출력), `preprocessor.py` (정제, 폴드 안의 대체·표준화·인코딩, 분할), `simulator.py` (대체 데이터, `generate_alternative_data()`) |
| `src/` | `features.py` (씬파일러 규칙, 변수 목록), `train.py`, `evaluate.py`, `scoring.py`, `actionability_table.py`, `german.py` |
| `notebooks/` | 보고서·그림·점검 스크립트 (`src/`는 이것에 의존하지 않음) |
| `tests/` | pytest 테스트 |
| `models/` | `xgboost_v1.0.joblib` (`{알고리즘}_v{model_version}.joblib`) |
| `reports/` | 보고서 (한국어, 영어), 그림, 결과표, 조치 가능성 Excel, 팀 인계 파일 |
| `mlruns/` | MLflow 기록 (커밋하지 않음, `src.train`으로 다시 생성) |

## 보고서

| 주제 | 보고서 |
|---|---|
| 데이터 정제와 씬파일러 정의 | `reports/data_report_ko.md` |
| 탐색적 분석 | `reports/eda_report_ko.md` |
| 대체 데이터 시뮬레이터 | `reports/simulator_report_ko.md` |
| 모델 비교, 테스트 결과, SHAP, 점수 체계 | `reports/model_report_ko.md` |
| 점검 (경로, 공정성, 안정성, 민감도) | `reports/audit_report_ko.md` |
| MLflow 안내 | `reports/mlflow_schema_ko.md` |
| ANOVA 인계 | `reports/anova_handoff_ko.md` |
| German Credit 보조 모델 | `reports/german_report_ko.md` |
| 조치 가능성 표 | `reports/actionability_table_ko.xlsx` (YAML에서 생성, 직접 수정 금지) |

모든 보고서는 영어 버전(`_en`)도 있습니다.

## 팀원 연계

**모두:** `data/processed/split.csv` (row_id, split)가 train / validation / test를 정합니다. 기준점과 공정성 완화는 **validation**에서 조정하고 **test**에서 보고합니다.

**체민규 (공정성, SHAP, DiCE, 비용 함수, ANOVA)**
- `config/actionability.yaml`과 Excel: DiCE가 바꿀 수 있는 변수(`dice_vary`), 방향, 단위, 허용 범위, 난이도, 단계당 개월. `spending_consistency`의 단위, 난이도, 기간은 멘토 검토가 필요한 팀 가정입니다.
- 경로 목표: 점수 495 (`recourse_target_score`), 승인은 475 유지. 495를 목표로 한 경로는 재학습해도 90.8%가 승인을 유지합니다.
- **거절 사유:** 권장 불가 변수 (SHAP 1위·2위인 `regular_payment_count`, `app_login_frequency`와 대출·신용한도 수, 부동산 담보대출 수)는 조언이 아니라 "참고(변경 불가)"로만 표시합니다. 시간 의존 경로는 기다리는 기간으로 표현합니다.
- **공정성 미결 과제:** 연령대 균등 기회(EO) 격차가 0.121로 목표 0.10을 충족하지 못합니다 (DI 0.832와 성별은 통과). 완화 작업은 체민규 담당입니다.
- **경로 미결 과제:** 495 목표의 경로 커버리지가 12개월 87.9% (24개월 91.8%)로 목표 90%에 미달합니다 (점검 보고서 참고). DiCE 커버리지는 체민규 담당입니다.
- DiCE에서 소득과 부채비율을 연결해야 합니다 (소득이 오르면 부채비율이 내려감).
- ANOVA 데이터: `reports/cv_fold_auc.csv` (5 × 5 폴드, 세그먼트 all / thin / general)와 `reports/anova_handoff_ko.md`.
- 실제 성별 점검: `reports/german/oof_predictions.csv` (`reports/german_report_ko.md` 참고).

**윤제진 (MLflow, Docker, FastAPI, Streamlit)**
- `reports/mlflow_schema_ko.md`: 실험 (알고리즘별, final, audit), 지표 키, 입력 열 21개, 모델 불러오는 방법. 모든 모델 아티팩트에 signature와 input example이 있으며, 서빙 모델은 확률을 반환합니다 (1번 열 = PD).
- docker-compose에서는 `MLFLOW_TRACKING_URI`를 MLflow 서버로 설정하고 학습을 다시 실행하십시오 (로컬 저장소는 절대 아티팩트 경로를 저장). 모델 레지스트리는 윤제진 담당입니다.
- 버전 포함 모델 파일: `models/xgboost_v1.0.joblib` (버전은 `config/train.yaml`, `/health`용으로 `final_summary.json`에도 있음).
- Docker와 CI에서 `REQUIRE_DATA=1`을 설정하십시오.

## 한계

- **시뮬레이션 편향:** 미션 예시처럼 시뮬레이터가 테스트 세트를 포함한 모든 행에 타깃을 사용합니다. GMSC만 사용한 결과가 현실적인 기준입니다.
- **상관 규칙:** 미션 변수 5개는 관측 r = −0.32로 보정됩니다 (체크리스트는 `df.corr()`로 확인). 그래서 각각이 어떤 실제 신용 변수보다 강합니다. r = −0.50에서는 변수 간 0.60 상한을 지킬 수 없습니다.
- **공정성:** 연령대 EO 격차 (0.121)가 목표 0.10을 충족하지 못하며, 재학습 시 기준점 근처 결정의 26.8%가 뒤바뀝니다.
- **PSI**는 같은 모집단의 무작위 표본을 비교하므로 구조적으로 0에 가깝고, GMSC에는 drift 검증용 날짜가 없습니다.
- **미션의 씬파일러 규칙** (연체 컬럼 2개 이상 결측 또는 카드 이력 12개월 미만)은 원본 GMSC에서 0%를 판정하므로 대리 기준을 사용합니다 (데이터 보고서 참고).
- 등급 기준, 승인 규칙, 경로 비용은 금융기관 값이 아닌 팀의 가정입니다.
- German Credit은 별도의 작은 모델이며 (1,000행, 미혼 여성 없음) GMSC와 수치를 비교할 수 없습니다.

## 데이터 출처

German Credit: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0). GMSC: Kaggle "Give Me Some Credit".
