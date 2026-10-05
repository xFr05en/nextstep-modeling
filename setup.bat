@echo off
REM Windows setup. Run from the project folder: setup.bat
python --version
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
python -c "import pandas, scipy, sklearn, lightgbm, xgboost, imblearn, mlflow, yaml; print('All packages OK')"
if exist data\raw\cs-training.csv (echo GMSC found) else (echo MISSING: put cs-training.csv in data\raw\)
if not exist .git git init -q
echo Done. Activate with: .venv\Scripts\activate
