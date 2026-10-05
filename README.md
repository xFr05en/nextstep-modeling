# NextStep: 데이터 · 시뮬레이터 · 모델링 (김원빈 파트)

> 본 시스템은 교육 목적으로 개발되었으며, 실제 금융 의사결정에 사용할 수 없습니다.

## 개요
XAI 기반 대안신용평가 시스템(4무원)의 데이터 로드, 대안데이터 시뮬레이터, 변수별 조치 가능성 메타데이터, 단조성 제약 모델 학습을 담당하는 코드입니다.

## 실행 방법
1. `data/raw/`에 `cs-training.csv`(GMSC)와 German Credit 파일을 넣습니다.
2. macOS/Linux: `bash setup.sh` / Windows: `setup.bat`
3. 가상환경을 활성화한 뒤 작업합니다.

## 폴더 구조
| 폴더 | 내용 |
|---|---|
| `config/` | `actionability.yaml`(조치 가능성 메타데이터), `simulator.yaml`(시뮬레이터 설정) |
| `data/` | 원본·전처리 데이터 (Git 제외) |
| `src/` | 데이터 처리, 시뮬레이터, 학습, 평가 코드 |
| `tests/` | pytest 테스트 |
| `models/` | 학습된 모델 |
| `reports/` | 결과표, 그림 |

## 다른 파트와의 연결
- 채민규: `config/actionability.yaml`과 학습된 모델 사용 (DiCE, 공정성)
- 윤제진: 6주차 팀 레포에 본 구조 그대로 병합 (MLflow, API)

## 상태
환경 설정 완료. 일부 결정 사항은 10/08 회의 후 확정 예정 (`CLAUDE.md` 참고).
