# NextStep: Data, Simulator, Modeling (Wonbin Kim's part)

> This system was developed for educational purposes and must not be used for real financial decisions. The alternative-data variables and gender are simulated, not real.

Part of team 4무원's XAI-based alternative credit scoring system for thin-filers (Capstone Design II / Industry Project, CSE4187, Sogang University). This part covers data cleaning, the alternative-data simulator, per-variable actionability metadata, model training with monotonic constraints, and the PD to credit score mapping.

## How to run

1. Put `cs-training.csv` (Kaggle "Give Me Some Credit") in `data/raw/`. For the secondary German Credit model, also put `german.data` there (download from the [UCI repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)).
2. Setup: macOS/Linux `bash setup.sh`, Windows `setup.bat`. Then `source .venv/bin/activate`.
3. Run the pipeline in order (about 4 minutes in total, fully seeded):

```bash
python -m src.data
```
```bash
python -m src.data.simulator
```
```bash
python -m src.train --stage all
```

4. Optional reports and figures:

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

5. Optional, German Credit secondary model (separate from GMSC, about 20 seconds):

```bash
python -m src.german
```

## Tests

Run from the project root (about 10 seconds):

```bash
pytest
```

Some tests need files that are not committed (`data/processed/gmsc_clean.csv`, `data/processed/gmsc_sim.csv`). Create them with the pipeline commands above.

- **Default:** if those files are missing, the tests that need them are **skipped** with a message saying which command to run.
- **`REQUIRE_DATA=1`:** missing files make those tests **fail** instead. Set this in Docker and CI so missing data can never pass silently:

```bash
REQUIRE_DATA=1 pytest
```

In a Dockerfile: `ENV REQUIRE_DATA=1`. In GitHub Actions: `env: REQUIRE_DATA: "1"` on the test step.

## Results (final model: XGBoost, all 19 features, no resampling, 11 monotonic constraints)

| Charter target | Result |
|---|---|
| AUC ≥ 0.78 | 0.927 ± 0.004 |
| KS ≥ 0.28 | 0.704 |
| Thin-filer AUC gain ≥ +0.03 | +0.071 |
| PSI < 0.1 | 0.0005 |
| AUC loss from monotonic constraints ≤ 0.01 | 0.0001 |
| pytest: at least 10 tests, 80% passing | 36 tests, 100% passing |

These numbers are optimistic because the alternative data is simulated from the real outcome. The realistic reference is the GMSC-only AUC of 0.865. See the limitations below.

## Configuration (all settings live here, not in code)

| File | Controls |
|---|---|
| `config/data.yaml` | Cleaning rules (special codes, utilization threshold, minimum age) |
| `config/simulator.yaml` | Alternative variables: distributions, latent correlation targets, credit link, thin-filer rule |
| `config/actionability.yaml` | Per-variable actionability class, direction, step, bounds, monotone sign, notes (Korean and English) |
| `config/train.yaml` | Model grid, hyperparameters, folds, winner rule, MLflow names |
| `config/scoring.yaml` | PD to score (0 to 1000), grades A to E, approval rule |

## Folder layout

| Folder | Contents |
|---|---|
| `config/` | The five YAML files above |
| `data/` | `raw/` and `processed/` (not committed) |
| `src/` | `data.py`, `simulator.py`, `features.py`, `train.py`, `evaluate.py`, `scoring.py`, `actionability_table.py` |
| `notebooks/` | Report and figure scripts (nothing in `src/` depends on them) |
| `tests/` | pytest suite |
| `models/` | `xgboost_v1.0.joblib` (versioned: `{algorithm}_v{model_version}.joblib`, version in `config/train.yaml`) |
| `reports/` | Step reports (Korean and English), figures, result tables, actionability Excel |
| `mlruns/` | MLflow tracking (SQLite, not committed; rebuilt by `src.train`) |

## Reports

| Step | Report |
|---|---|
| 1. Data cleaning | `reports/data_report_en.md` |
| 2. EDA | `reports/eda_report_en.md` |
| 3. Simulator and thin-filer flag | `reports/simulator_report_en.md` |
| 5. Model comparison, constraints, scoring | `reports/model_report_en.md` |
| MLflow guide | `reports/mlflow_schema_en.md` |
| Actionability table | `reports/actionability_table_en.xlsx` (generated from the YAML; never edit by hand) |
| Audit (recourse, fairness, stability, holdout, sensitivity) | `reports/audit_report_en.md` |
| German Credit secondary model | `reports/german_report_en.md` |

Every report also has a Korean version (`_ko`).

## For teammates

**채민규 (fairness, SHAP, DiCE, cost function)**
- `config/actionability.yaml`: which variables DiCE may vary (`dice_vary`), direction, step, bounds, difficulty, months per step. The Excel files show the same content.
- Recourse target: `recourse_target_score` = 495 in `config/scoring.yaml`. Approval itself stays at grades A to C (score 475 or higher); the 20-point margin keeps paths approved when the model is retrained (see `reports/audit_report_en.md`). Use `src/scoring.py` to map PD to score, grade and approval.
- Autopay is a model feature but is not used in paths: class NOT_RECOMMENDED, `dice_vary: false`, monotone -1. In SHAP rejection reasons, list autopay under "reference (not changeable)", not under actionable reasons, so the explanation never suggests setting up autopay.
- Model: `models/xgboost_v1.0.joblib`, a pipeline that takes the 21 input columns as a DataFrame. `predict_proba[:, 1]` is the PD.
- Fairness inputs: `gender_female` (simulated, protected, not a model feature) and `age` (protected, used as a feature). Age correlates with utilization and dependents (see the EDA and simulator reports).
- Real-gender check: `reports/german/oof_predictions.csv` has out-of-fold PD (XGBoost and LR), actual outcome, sex, age, personal status and foreign worker for the 1,000 German Credit applicants. No cutoff is applied; note the UCI cost matrix (break-even PD about 0.167) in `reports/german_report_en.md`.

**윤제진 (MLflow, Docker, FastAPI, Streamlit)**
- `reports/mlflow_schema_en.md`: experiment and run names, metric keys, input columns, how to load the model. The served model returns probabilities (column 1 = PD).
- MLflow uses SQLite in `mlruns/mlflow.db`, because MLflow 3.16 no longer accepts the plain folder backend. Browse it with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- Set `REQUIRE_DATA=1` in Docker and CI (see Tests).

## Limitations

- **Simulation bias.** The alternative variables are generated from the real default column, so models that use them look better than real data would allow. The trade-off patterns (for example, how the credit link lowers the gain) are more reliable than the absolute numbers.
- **The correlation rule is applied on the copula's latent scale** (0.35 for all five variables). Observed Pearson correlations with default are about 0.18 to 0.27.
- **PSI uses random folds**, so it is close to 0 by construction. GMSC has no dates, so drift over time cannot be tested.
- **Policy choices are assumptions:** grade cutoffs, the approval rule, time per step and difficulty are team assumptions, not values from a lender.
- German Credit is a separate small model (1,000 rows, no single women in the data); its numbers are not comparable with GMSC. See `reports/german_report_en.md`.

## Data credit

German Credit: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0). GMSC: Kaggle "Give Me Some Credit".
