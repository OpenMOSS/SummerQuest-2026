#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_RUN="${PYTHON_RUN:-.venv/bin/python}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-.tiktoken-cache}"

RUN_TINYSTORIES="${RUN_TINYSTORIES:-1}"
RUN_OWT_ENCODE="${RUN_OWT_ENCODE:-1}"
RUN_OWT_TRAIN="${RUN_OWT_TRAIN:-1}"

DEVICE="${DEVICE:-cuda}"

TINYSTORIES_SCRIPT="${TINYSTORIES_SCRIPT:-scripts/run_tinystories_gpu_experiments.sh}"
OWT_SCRIPT="${OWT_SCRIPT:-scripts/run_owt_gpu_full_experiment.sh}"

OWT_RUN_NAME="${OWT_RUN_NAME:-owt_gpu_full}"
OWT_TOKENIZER_RUN_NAME="${OWT_TOKENIZER_RUN_NAME:-owt_main_cpu_full_tokenizer}"
OWT_TRAIN_TEXT="${OWT_TRAIN_TEXT:-data/owt_train.txt}"
OWT_VALID_TEXT="${OWT_VALID_TEXT:-data/owt_valid.txt}"
OWT_TOKENIZER_TRAIN_TEXT="${OWT_TOKENIZER_TRAIN_TEXT:-$OWT_TRAIN_TEXT}"
OWT_VOCAB_SIZE="${OWT_VOCAB_SIZE:-32000}"

OWT_VOCAB_PATH="${OWT_VOCAB_PATH:-runs/tokenizers/${OWT_TOKENIZER_RUN_NAME}_vocab.json}"
OWT_MERGES_PATH="${OWT_MERGES_PATH:-runs/tokenizers/${OWT_TOKENIZER_RUN_NAME}_merges.txt}"
OWT_TRAIN_BIN="${OWT_TRAIN_BIN:-runs/encoded/${OWT_TOKENIZER_RUN_NAME}_train.bin}"
OWT_VALID_BIN="${OWT_VALID_BIN:-runs/encoded/${OWT_TOKENIZER_RUN_NAME}_valid.bin}"

FORCE_OWT_TOKENIZER="${FORCE_OWT_TOKENIZER:-0}"
FORCE_OWT_TRAIN_ENCODE="${FORCE_OWT_TRAIN_ENCODE:-0}"
FORCE_OWT_VALID_ENCODE="${FORCE_OWT_VALID_ENCODE:-1}"
ALLOW_HUGE_TOKENIZER="${ALLOW_HUGE_TOKENIZER:-0}"
MAX_TOKENIZER_BYTES="${MAX_TOKENIZER_BYTES:-1073741824}"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

check_runtime() {
  require_file "$PYTHON_RUN"
  require_file "$TINYSTORIES_SCRIPT"
  require_file "$OWT_SCRIPT"
  "$PYTHON_RUN" - <<PY
import sys
import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))

if "$DEVICE" == "cuda" and not torch.cuda.is_available():
    raise SystemExit("DEVICE=cuda requested, but torch.cuda.is_available() is False")
PY
}

train_owt_tokenizer_if_needed() {
  if [[ "$FORCE_OWT_TOKENIZER" != "1" && -f "$OWT_VOCAB_PATH" && -f "$OWT_MERGES_PATH" ]]; then
    log "skip OWT tokenizer: existing vocab=$OWT_VOCAB_PATH merges=$OWT_MERGES_PATH"
    return 0
  fi

  require_file "$OWT_TOKENIZER_TRAIN_TEXT"
  mkdir -p "$(dirname "$OWT_VOCAB_PATH")" "$(dirname "$OWT_MERGES_PATH")"

  local tokenizer_bytes
  tokenizer_bytes="$(wc -c < "$OWT_TOKENIZER_TRAIN_TEXT")"
  if [[ "$tokenizer_bytes" -gt "$MAX_TOKENIZER_BYTES" && "$ALLOW_HUGE_TOKENIZER" != "1" ]]; then
    cat >&2 <<EOF
Refusing to train OWT tokenizer on a very large text file.

OWT_TOKENIZER_TRAIN_TEXT=$OWT_TOKENIZER_TRAIN_TEXT
size_bytes=$tokenizer_bytes
MAX_TOKENIZER_BYTES=$MAX_TOKENIZER_BYTES

Reuse existing OWT_VOCAB_PATH/OWT_MERGES_PATH, set OWT_TOKENIZER_TRAIN_TEXT to a smaller corpus,
or set ALLOW_HUGE_TOKENIZER=1 if you intentionally want full OWT tokenizer training.
EOF
    exit 1
  fi

  log "train OWT tokenizer: input=$OWT_TOKENIZER_TRAIN_TEXT vocab_size=$OWT_VOCAB_SIZE"
  "$PYTHON_RUN" scripts/train_tokenizer.py \
    --input "$OWT_TOKENIZER_TRAIN_TEXT" \
    --vocab-size "$OWT_VOCAB_SIZE" \
    --vocab-out "$OWT_VOCAB_PATH" \
    --merges-out "$OWT_MERGES_PATH"
  log "OWT tokenizer done"
}

encode_owt_split() {
  local split="$1"
  local input="$2"
  local output="$3"
  local force="$4"

  require_file "$input"
  require_file "$OWT_VOCAB_PATH"
  require_file "$OWT_MERGES_PATH"
  mkdir -p "$(dirname "$output")"

  if [[ "$force" != "1" && -f "$output" ]]; then
    log "skip OWT $split encoding: existing $output"
    return 0
  fi

  local tmp_output
  tmp_output="${output}.tmp.$$"
  rm -f "$tmp_output"
  log "encode OWT $split split: input=$input output=$output"
  "$PYTHON_RUN" scripts/encode.py \
    --input "$input" \
    --output "$tmp_output" \
    --vocab "$OWT_VOCAB_PATH" \
    --merges "$OWT_MERGES_PATH" \
    --special-token '<|endoftext|>'
  mv "$tmp_output" "$output"
  log "OWT $split encoding done: $output"
}

run_tinystories_stage() {
  if [[ "$RUN_TINYSTORIES" != "1" ]]; then
    log "skip TinyStories stage: RUN_TINYSTORIES=$RUN_TINYSTORIES"
    return 0
  fi

  log "stage 1/3: TinyStories required GPU experiments"
  DEVICE="$DEVICE" "$TINYSTORIES_SCRIPT"
  log "stage 1/3 done"
}

run_owt_encode_stage() {
  if [[ "$RUN_OWT_ENCODE" != "1" ]]; then
    log "skip OWT encoding stage: RUN_OWT_ENCODE=$RUN_OWT_ENCODE"
    return 0
  fi

  log "stage 2/3: OWT tokenizer and encoding"
  train_owt_tokenizer_if_needed
  encode_owt_split train "$OWT_TRAIN_TEXT" "$OWT_TRAIN_BIN" "$FORCE_OWT_TRAIN_ENCODE"
  encode_owt_split valid "$OWT_VALID_TEXT" "$OWT_VALID_BIN" "$FORCE_OWT_VALID_ENCODE"
  log "stage 2/3 done"
}

run_owt_train_stage() {
  if [[ "$RUN_OWT_TRAIN" != "1" ]]; then
    log "skip OWT training stage: RUN_OWT_TRAIN=$RUN_OWT_TRAIN"
    return 0
  fi

  require_file "$OWT_TRAIN_BIN"
  require_file "$OWT_VALID_BIN"

  log "stage 3/3: OWT full-model GPU training"
  env \
    RUN_NAME="$OWT_RUN_NAME" \
    TOKENIZER_RUN_NAME="$OWT_TOKENIZER_RUN_NAME" \
    DEVICE="$DEVICE" \
    TRAIN_TEXT="$OWT_TRAIN_TEXT" \
    VALID_TEXT="$OWT_VALID_TEXT" \
    TOKENIZER_TRAIN_TEXT="$OWT_TOKENIZER_TRAIN_TEXT" \
    VOCAB_SIZE="$OWT_VOCAB_SIZE" \
    VOCAB_PATH="$OWT_VOCAB_PATH" \
    MERGES_PATH="$OWT_MERGES_PATH" \
    TRAIN_BIN="$OWT_TRAIN_BIN" \
    VALID_BIN="$OWT_VALID_BIN" \
    FORCE_TOKENIZER=0 \
    FORCE_ENCODE=0 \
    "$OWT_SCRIPT"
  log "stage 3/3 done"
}

main() {
  mkdir -p runs/tokenizers runs/encoded runs/checkpoints runs/logs runs/configs runs/experiment_manifests
  check_runtime
  run_tinystories_stage
  run_owt_encode_stage
  run_owt_train_stage
}

main "$@"
