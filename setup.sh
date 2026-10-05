#!/usr/bin/env bash
# macOS / Linux setup. Run from the project folder:  bash setup.sh
set -e
PY=${PYTHON:-python3}
echo "Python: $($PY --version)"
$PY -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null; then
  brew list libomp >/dev/null 2>&1 || { echo "Installing libomp (needed by LightGBM/XGBoost on macOS)"; brew install libomp; }
fi
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
python -c "import pandas, scipy, sklearn, lightgbm, xgboost, imblearn, mlflow, yaml; print('All packages OK')"
[ -f data/raw/cs-training.csv ] && echo "GMSC found" || echo "MISSING: put cs-training.csv in data/raw/"
[ -d .git ] || { git init -q && echo "git repo initialized"; }
echo "Done. Activate with: source .venv/bin/activate"
