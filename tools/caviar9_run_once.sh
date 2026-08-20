#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/cuda/bin/python}"
DATA_DIR="${CODEX_GPU_DATA_DIR:-$HOME/codex-gpu-work/data}"

exec "$PYTHON_BIN" tools/caviar9_git_runner.py \
  --python-bin "$PYTHON_BIN" \
  --data-dir "$DATA_DIR" \
  "$@"
