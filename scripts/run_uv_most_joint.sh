#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

uvgen_python="${UVGEN_PYTHON:-python3}"
if [[ ! -x .uvgen_venv/bin/python ]]; then
  "$uvgen_python" -m venv .uvgen_venv
fi

.uvgen_venv/bin/python -m pip install -r requirements_uv_most_joint.txt
exec .uvgen_venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  UV_MOST_Joint_RL.ipynb \
  --ExecutePreprocessor.timeout=7200
