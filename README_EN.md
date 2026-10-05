# NextStep: Data, Simulator, Modeling (Wonbin Kim's part)

> This system was developed for educational purposes and must not be used for real financial decisions.

## Overview
Code for data loading, the alternative-data simulator, per-variable actionability metadata, and monotonic-constraint model training for team 4무원's XAI-based alternative credit scoring system.

## How to run
1. Put `cs-training.csv` (GMSC) and the German Credit file in `data/raw/`.
2. macOS/Linux: `bash setup.sh` / Windows: `setup.bat`
3. Activate the virtual environment and work from there.

## Folder layout
| Folder | Contents |
|---|---|
| `config/` | `actionability.yaml` (actionability metadata), `simulator.yaml` (simulator settings) |
| `data/` | Raw and processed data (not committed) |
| `src/` | Data processing, simulator, training, evaluation |
| `tests/` | pytest tests |
| `models/` | Trained models |
| `reports/` | Results tables, figures |

## Tests
Run from the project root (about 10 seconds):

```bash
pytest
```

Some tests need files that are not committed (`data/processed/gmsc_clean.csv`, `data/processed/gmsc_sim.csv`). Create them with `python -m src.data`, `python -m src.simulator` and `python -m src.train --stage all`.

- **Default:** if those files are missing, the tests that need them are **skipped** with a message saying which command to run.
- **`REQUIRE_DATA=1`:** missing files make those tests **fail** instead. Set this in Docker and CI so missing data can never pass silently:

```bash
REQUIRE_DATA=1 pytest
```

In a Dockerfile: `ENV REQUIRE_DATA=1`. In GitHub Actions: `env: REQUIRE_DATA: "1"` on the test step.

## Links to other parts
- 채민규: uses `config/actionability.yaml` and the trained model (DiCE, fairness)
- 윤제진: merges this structure into the team repo in week 6 (MLflow, API)

## Status
Environment ready. Some decisions will be finalized after the 10/08 meeting (see `CLAUDE.md`).
