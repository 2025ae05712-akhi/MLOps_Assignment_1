#!/usr/bin/env bash
# Reproduces the pipeline on a machine with Python 3.10+ (installs Python 3.11 via uv).
set -euo pipefail
cd "$(dirname "$0")"

for tool in python3 git docker kubectl; do
  if command -v "$tool" >/dev/null 2>&1; then echo "found:   $tool"; else echo "missing: $tool"; fi
done

python3 -m pip install --user -q uv
python3 -m uv venv --python 3.11 .venv
python3 -m uv pip install --python .venv/bin/python -q -r requirements-dev.txt

export MPLBACKEND=Agg
.venv/bin/python data/download_data.py
.venv/bin/ruff check .
.venv/bin/python -m scripts.eda
.venv/bin/python -m src.train
.venv/bin/python -m pytest -q

echo "Pipeline OK. Start the API with: .venv/bin/uvicorn src.api:app --port 8000"
echo "View MLflow with: .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
