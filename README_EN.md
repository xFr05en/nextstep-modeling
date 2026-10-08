# NextStep: Data, Simulator, Modeling (Wonbin Kim's part)

> This system was developed for educational purposes and must not be used for real financial decisions. The alternative-data variables and gender are simulated, not real.

Part of team 4무원's XAI-based alternative credit scoring system for thin-filers (Capstone Design II / Industry Project, CSE4187, Sogang University). This part covers data loading and cleaning, the alternative-data simulator, per-variable actionability metadata, model training with monotonic constraints, and the PD to credit score mapping.

## How to run

1. Put `cs-training.csv` (Kaggle "Give Me Some Credit") in `data/raw/`. For the secondary German Credit model, also put `german.data` there ([UCI repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)).
2. Setup: macOS/Linux `bash setup.sh`, Windows `setup.bat`. Then `source .venv/bin/activate`.
3. Run the pipeline in order (about 15 minutes, fully seeded):

```bash
python -m src.data
```
```bash
python -m src.data.simulator
```
```bash
python -m src.train --stage all
```

4. Optional reports, figures and checks:

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

5. Optional, German Credit secondary model (separate from GMSC, about 20 seconds):

```bash
python -m src.german
```

## Tests

Run from the project root (about 30 seconds, 91 tests):

```bash
pytest
```

Some tests need files that are not committed (`data/processed/*.csv`, `mlruns/`). Create them with the pipeline commands above.

- **Default:** if those files are missing, the tests that need them are **skipped** with a message saying which command to run.
- **`REQUIRE_DATA=1`:** missing files make those tests **fail** instead. Set this in Docker and CI:

```bash
REQUIRE_DATA=1 pytest
```

In a Dockerfile: `ENV REQUIRE_DATA=1`. In GitHub Actions: `env: REQUIRE_DATA: "1"` on the test step.

## Results (official: test set, shipped model `models/xgboost_v1.0.joblib`)

XGBoost, 21 features, no resampling, 13 monotonic constraints; trained on train + validation, evaluated once on the 15% test set. 95% bootstrap intervals.

| Charter target | Test result |
|---|---|
| AUC ≥ 0.78 | 0.949 (0.944 to 0.953) |
| KS ≥ 0.28 | 0.760 (0.745 to 0.776) |
| Thin-filer AUC gain ≥ +0.03 | +0.071 (+0.052 to +0.092) |
| PSI < 0.1 | 0.0004 |
| AUC loss from monotonic constraints ≤ 0.01 | 0.0000 (validation) |
| pytest: at least 10 tests, 80% passing | 91 tests, 100% passing |

**These numbers are optimistic.** The simulator uses the real outcome for every row, including the test set (as in the mission's example), and each mission variable alone reaches AUC 0.80 to 0.84. The realistic reference is the GMSC-only model (CV AUC 0.864; thin-filer test AUC 0.861). See `reports/model_report_en.md`.

## Configuration (all settings live here, not in code)

| File | Controls |
|---|---|
| `config/data.yaml` | Cleaning rules (special codes, utilization threshold, minimum age) |
| `config/simulator.yaml` | The 8 alternative variables (distributions, observed / latent correlation targets, shared factor), credit link, `thin_filer_ratio` / `thin_filer_mode` / `bias_ratio`, thin-filer rules |
| `config/actionability.yaml` | Per-variable actionability class, direction, step, bounds, monotone sign, `dice_vary`, `model_feature`, notes (Korean and English) |
| `config/train.yaml` | Split (70/15/15), CV (5 × 5-fold), model grid, hyperparameters, winner rule, model version, MLflow experiments |
| `config/scoring.yaml` | PD to score (0 to 1000), grades A to E, approval rule (score ≥ 475), recourse target (495) |
| `config/german.yaml` | German Credit secondary model |
| `.env.example` | `MLFLOW_TRACKING_URI` (local `./mlruns` by default; the compose MLflow server in Docker) |

## Folder layout

| Folder | Contents |
|---|---|
| `config/` | The YAML files above |
| `data/` | `raw/` and `processed/` (not committed); `processed/split.csv` defines train / validation / test |
| `src/data/` | `loader.py` (loads GMSC / German Credit, prints column names, dtypes, missing ratios), `preprocessor.py` (cleaning, in-fold imputer / scaler / encoder, split), `simulator.py` (alternative data, `generate_alternative_data()`) |
| `src/` | `features.py` (thin-filer rules, feature lists), `train.py`, `evaluate.py`, `scoring.py`, `actionability_table.py`, `german.py` |
| `notebooks/` | Report, figure and audit scripts (nothing in `src/` depends on them) |
| `tests/` | pytest suite |
| `models/` | `xgboost_v1.0.joblib` (`{algorithm}_v{model_version}.joblib`) |
| `reports/` | Reports (Korean and English), figures, result tables, actionability Excel, team handoff files |
| `mlruns/` | MLflow tracking (not committed; rebuilt by `src.train`) |

## Reports

| Topic | Report |
|---|---|
| Data cleaning and thin-filer definition | `reports/data_report_en.md` |
| EDA | `reports/eda_report_en.md` |
| Alternative-data simulator | `reports/simulator_report_en.md` |
| Model comparison, test results, SHAP, scoring | `reports/model_report_en.md` |
| Audit (recourse, fairness, stability, sensitivity) | `reports/audit_report_en.md` |
| MLflow guide | `reports/mlflow_schema_en.md` |
| ANOVA handoff | `reports/anova_handoff_en.md` |
| German Credit secondary model | `reports/german_report_en.md` |
| Actionability table | `reports/actionability_table_en.xlsx` (generated from the YAML; never edit by hand) |

Every report also has a Korean version (`_ko`).

## For teammates

**Everyone:** `data/processed/split.csv` (row_id, split) defines train / validation / test. Tune thresholds and fairness mitigation on **validation**; report them on **test**.

**체민규 (fairness, SHAP, DiCE, cost function, ANOVA)**
- `config/actionability.yaml` and the Excel files: which variables DiCE may vary (`dice_vary`), direction, step, bounds, difficulty, months per step. `spending_consistency`'s step, difficulty and duration are team assumptions that need mentor review.
- Recourse target: score 495 (`recourse_target_score`); approval stays at 475. Paths aimed at 495 stay approved 90.8% of the time under retraining.
- **Rejection reasons:** the NOT_RECOMMENDED variables (`regular_payment_count`, `app_login_frequency`, which are SHAP #1 and #2, plus open credit lines and real estate loans) appear only under "reference (not changeable)", never as advice. TIME_ONLY paths are phrased as waiting time.
- **Open fairness item:** the age-band equalized-odds gap is 0.121, which fails the 0.10 target (DI 0.832 passes; gender passes). Mitigation is yours.
- **Open recourse item:** path coverage at the 495 target is 87.9% over 12 months (91.8% over 24 months), below the 90% goal (see the audit report). DiCE coverage is yours.
- DiCE should link income and debt ratio (raising income lowers the ratio).
- ANOVA data: `reports/cv_fold_auc.csv` (5 × 5 folds, segments all / thin / general) and `reports/anova_handoff_en.md`.
- Real-gender check: `reports/german/oof_predictions.csv` (see `reports/german_report_en.md`).

**윤제진 (MLflow, Docker, FastAPI, Streamlit)**
- `reports/mlflow_schema_en.md`: experiments (one per algorithm, plus final and audit), metric keys, the 21 input columns, how to load the model. Every model artifact has a signature and input example; the served model returns probabilities (column 1 = PD).
- In docker-compose, set `MLFLOW_TRACKING_URI` to the MLflow server and rerun training (the local store saves absolute artifact paths). The Model Registry is yours.
- Versioned model file: `models/xgboost_v1.0.joblib` (version in `config/train.yaml`; also in `final_summary.json` for `/health`).
- Set `REQUIRE_DATA=1` in Docker and CI.

## Limitations

- **Simulation bias:** the simulator uses the target for all rows, test included, as in the mission's example. The GMSC-only results are the realistic reference.
- **Correlation rule:** the 5 mission variables are calibrated to observed r = −0.32 (the checklist verifies `df.corr()`), which makes each of them stronger than any real credit feature. At r = −0.50 the 0.60 cap between variables cannot be met.
- **Fairness:** the age-band EO gap (0.121) fails the 0.10 target; decisions near the cutoff flip in 26.8% of cases when retrained.
- **PSI** compares random samples of the same population, so it is close to 0 by construction; GMSC has no dates for a drift test.
- **The mission's thin-filer rule** (2+ missing past-due columns or card history < 12 months) flags 0% of raw GMSC, so a proxy is used (see the data report).
- Grade cutoffs, approval rule and recourse costs are team assumptions, not values from a lender.
- German Credit is a separate small model (1,000 rows, no single women); its numbers are not comparable with GMSC.

## Data credit

German Credit: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 (CC BY 4.0). GMSC: Kaggle "Give Me Some Credit".
