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
TRAIN_TXT="${TRAIN_TXT:-$DATA_DIR/TinyStoriesV2-GPT4-train.txt}"
VAL_TXT="${VAL_TXT:-$DATA_DIR/TinyStoriesV2-GPT4-valid.txt}"
TOKENIZER_OUT="${TOKENIZER_OUT:-$DATA_DIR/tinystories_tokenizer.pkl}"
TRAIN_NPY="${TRAIN_NPY:-$DATA_DIR/tinystories_train.npy}"
VAL_NPY="${VAL_NPY:-$DATA_DIR/tinystories_val.npy}"
VOCAB_SIZE="${VOCAB_SIZE:-10000}"
DTYPE="${DTYPE:-uint16}"
TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-40}"
ENCODE_WORKERS="${ENCODE_WORKERS:-40}"

mkdir -p "$DATA_DIR"

if [[ ! -f "$TRAIN_TXT" || ! -f "$VAL_TXT" ]]; then
  echo "missing TinyStories txt files; run scripts/download_a1_data.sh first" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER_OUT" ]]; then
  "$PYTHON" scripts/train_tokenizer.py \
    --input "$TRAIN_TXT" \
    --output "$TOKENIZER_OUT" \
    --vocab-size "$VOCAB_SIZE" \
    --special-token "<|endoftext|>" \
    --num-workers "$TOKENIZER_WORKERS" \
    --metadata "$DATA_DIR/tinystories_tokenizer_meta.json"
else
  echo "exists: $TOKENIZER_OUT"
fi

if [[ ! -f "$TRAIN_NPY" ]]; then
  "$PYTHON" scripts/encode_data.py \
    --input "$TRAIN_TXT" \
    --tokenizer "$TOKENIZER_OUT" \
    --output "$TRAIN_NPY" \
    --dtype "$DTYPE" \
    --num-workers "$ENCODE_WORKERS" \
    --split-mode special \
    --metadata "$DATA_DIR/tinystories_train_meta.json"
else
  echo "exists: $TRAIN_NPY"
fi

if [[ ! -f "$VAL_NPY" ]]; then
  "$PYTHON" scripts/encode_data.py \
    --input "$VAL_TXT" \
    --tokenizer "$TOKENIZER_OUT" \
    --output "$VAL_NPY" \
    --dtype "$DTYPE" \
    --num-workers "$ENCODE_WORKERS" \
    --split-mode special \
    --metadata "$DATA_DIR/tinystories_val_meta.json"
else
  echo "exists: $VAL_NPY"
fi

echo "done: TinyStories tokenizer and encoded arrays"
