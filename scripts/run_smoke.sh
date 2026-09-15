#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec ./venv/bin/python -m mostgen run-all --mode smoke --output runs/smoke

