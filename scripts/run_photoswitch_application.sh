#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ ! -x .uvgen_venv/bin/python ]]; then
  python3 -m venv .uvgen_venv
fi
.uvgen_venv/bin/python -m pip install -r requirements_uv_most_joint.txt
exec .uvgen_venv/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  UV_Photoswitch_Application_RL.ipynb \
  --ExecutePreprocessor.timeout=7200
