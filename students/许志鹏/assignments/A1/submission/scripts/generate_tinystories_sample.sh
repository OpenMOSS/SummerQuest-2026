#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv-cu128/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON_FALLBACK:-python}"
fi

DATA_DIR="${DATA_DIR:-data}"
RUN_DIR="${RUN_DIR:-runs/tinystories_baseline}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
PROMPT="${PROMPT:-Once upon a time}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-42}"

"$PYTHON" scripts/generate.py \
  --checkpoint "$RUN_DIR/ckpt_latest.pt" \
  --config "$RUN_DIR/config.json" \
  --tokenizer "$DATA_DIR/tinystories_tokenizer.pkl" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED"
