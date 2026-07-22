#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${A1_WORKDIR:-"$SCRIPT_DIR/.."}"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

mkdir -p job_logs artifacts runs/owt_gpu

TOKENIZER_PID="${1:-}"

echo "[owt-pipeline] started at $(date)"
echo "[owt-pipeline] tokenizer pid: ${TOKENIZER_PID:-none}"

if [[ -n "$TOKENIZER_PID" ]]; then
  while kill -0 "$TOKENIZER_PID" 2>/dev/null; do
    echo "[owt-pipeline] tokenizer still running at $(date)"
    sleep 300
  done
else
  while pgrep -f "train_tokenizer.py.*owt_train.txt" >/dev/null; do
    echo "[owt-pipeline] tokenizer still running at $(date)"
    sleep 300
  done
fi

echo "[owt-pipeline] tokenizer process ended at $(date)"

if [[ ! -s artifacts/owt_vocab.json || ! -s artifacts/owt_merges.txt ]]; then
  echo "[owt-pipeline] ERROR: tokenizer outputs missing or empty"
  ls -lh artifacts/owt_vocab.json artifacts/owt_merges.txt || true
  exit 1
fi

echo "[owt-pipeline] encoding OWT train at $(date)"
PYTHONPATH=. python scripts/encode_dataset.py \
  --input data/owt_train.txt \
  --vocab artifacts/owt_vocab.json \
  --merges artifacts/owt_merges.txt \
  --output artifacts/owt_train.npy \
  --dtype uint16

echo "[owt-pipeline] encoding OWT val at $(date)"
PYTHONPATH=. python scripts/encode_dataset.py \
  --input data/owt_valid.txt \
  --vocab artifacts/owt_vocab.json \
  --merges artifacts/owt_merges.txt \
  --output artifacts/owt_val.npy \
  --dtype uint16

echo "[owt-pipeline] starting OWT training at $(date)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_lm.py \
  --config configs/owt_gpu.json

echo "[owt-pipeline] finished at $(date)"
