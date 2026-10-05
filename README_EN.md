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

## Links to other parts
- 채민규: uses `config/actionability.yaml` and the trained model (DiCE, fairness)
- 윤제진: merges this structure into the team repo in week 6 (MLflow, API)

## Status
Environment ready. Some decisions will be finalized after the 10/08 meeting (see `CLAUDE.md`).
