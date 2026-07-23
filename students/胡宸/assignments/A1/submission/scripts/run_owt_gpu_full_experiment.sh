#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_RUN="${PYTHON_RUN:-.venv/bin/python}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-.tiktoken-cache}"

RUN_NAME="${RUN_NAME:-owt_gpu_full}"
TOKENIZER_RUN_NAME="${TOKENIZER_RUN_NAME:-owt_main_cpu_full_tokenizer}"
DEVICE="${DEVICE:-cuda}"

TRAIN_TEXT="${TRAIN_TEXT:-data/owt_train.txt}"
VALID_TEXT="${VALID_TEXT:-data/owt_valid.txt}"
TOKENIZER_TRAIN_TEXT="${TOKENIZER_TRAIN_TEXT:-$TRAIN_TEXT}"

VOCAB_SIZE="${VOCAB_SIZE:-32000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-16}"
D_FF="${D_FF:-1344}"
ROPE_THETA="${ROPE_THETA:-10000.0}"

BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-10000}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SEED="${SEED:-0}"
NUM_THREADS="${NUM_THREADS:-}"
TF32="${TF32:-1}"
COMPILE="${COMPILE:-0}"

FORCE_TOKENIZER="${FORCE_TOKENIZER:-0}"
FORCE_ENCODE="${FORCE_ENCODE:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
RESUME="${RESUME:-1}"
ALLOW_HUGE_TOKENIZER="${ALLOW_HUGE_TOKENIZER:-0}"
MAX_TOKENIZER_BYTES="${MAX_TOKENIZER_BYTES:-1073741824}"

VOCAB_PATH="${VOCAB_PATH:-runs/tokenizers/${TOKENIZER_RUN_NAME}_vocab.json}"
MERGES_PATH="${MERGES_PATH:-runs/tokenizers/${TOKENIZER_RUN_NAME}_merges.txt}"
TRAIN_BIN="${TRAIN_BIN:-runs/encoded/${TOKENIZER_RUN_NAME}_train.bin}"
VALID_BIN="${VALID_BIN:-runs/encoded/${TOKENIZER_RUN_NAME}_valid.bin}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/${RUN_NAME}.pt}"
TRAIN_LOG="${TRAIN_LOG:-runs/logs/${RUN_NAME}_train.jsonl}"
CONSOLE_LOG="${CONSOLE_LOG:-runs/logs/${RUN_NAME}.console.log}"
CONFIG_PATH="${CONFIG_PATH:-runs/configs/${RUN_NAME}.env}"

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

latest_step_checkpoint() {
  local output="$1"
  local stem suffix candidate latest_step latest_path step_part step
  stem="${output%.*}"
  suffix=".${output##*.}"
  latest_step=-1
  latest_path=""
  shopt -s nullglob
  for candidate in "${stem}"_step*"${suffix}"; do
    step_part="${candidate#${stem}_step}"
    step="${step_part%${suffix}}"
    if [[ "$step" =~ ^[0-9]+$ ]] && (( step > latest_step )); then
      latest_step="$step"
      latest_path="$candidate"
    fi
  done
  shopt -u nullglob
  printf '%s\n' "$latest_path"
}

check_runtime() {
  require_file "$PYTHON_RUN"
  require_file "$TRAIN_TEXT"
  require_file "$VALID_TEXT"

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

train_tokenizer_if_needed() {
  if [[ "$FORCE_TOKENIZER" != "1" && -f "$VOCAB_PATH" && -f "$MERGES_PATH" ]]; then
    log "skip tokenizer: existing vocab=$VOCAB_PATH merges=$MERGES_PATH"
    return 0
  fi

  require_file "$TOKENIZER_TRAIN_TEXT"

  local tokenizer_bytes
  tokenizer_bytes="$(wc -c < "$TOKENIZER_TRAIN_TEXT")"
  if [[ "$tokenizer_bytes" -gt "$MAX_TOKENIZER_BYTES" && "$ALLOW_HUGE_TOKENIZER" != "1" ]]; then
    cat >&2 <<EOF
Refusing to train the tokenizer on a very large OWT text file with the Python BPE implementation.

TOKENIZER_TRAIN_TEXT=$TOKENIZER_TRAIN_TEXT
size_bytes=$tokenizer_bytes
MAX_TOKENIZER_BYTES=$MAX_TOKENIZER_BYTES

Either reuse an existing tokenizer via VOCAB_PATH/MERGES_PATH, train on a smaller tokenizer corpus
with TOKENIZER_TRAIN_TEXT=..., or set ALLOW_HUGE_TOKENIZER=1 if you really want full OWT BPE training.
EOF
    exit 1
  fi

  log "train tokenizer: input=$TOKENIZER_TRAIN_TEXT vocab_size=$VOCAB_SIZE"
  "$PYTHON_RUN" scripts/train_tokenizer.py \
    --input "$TOKENIZER_TRAIN_TEXT" \
    --vocab-size "$VOCAB_SIZE" \
    --vocab-out "$VOCAB_PATH" \
    --merges-out "$MERGES_PATH"
  log "tokenizer done: vocab=$VOCAB_PATH merges=$MERGES_PATH"
}

encode_if_needed() {
  require_file "$VOCAB_PATH"
  require_file "$MERGES_PATH"

  if [[ "$FORCE_ENCODE" == "1" || ! -f "$TRAIN_BIN" ]]; then
    log "encode train split: input=$TRAIN_TEXT output=$TRAIN_BIN"
    "$PYTHON_RUN" scripts/encode.py \
      --input "$TRAIN_TEXT" \
      --output "$TRAIN_BIN" \
      --vocab "$VOCAB_PATH" \
      --merges "$MERGES_PATH" \
      --special-token '<|endoftext|>'
    log "train encoding done: $TRAIN_BIN"
  else
    log "skip train encoding: existing $TRAIN_BIN"
  fi

  if [[ "$FORCE_ENCODE" == "1" || ! -f "$VALID_BIN" ]]; then
    log "encode valid split: input=$VALID_TEXT output=$VALID_BIN"
    "$PYTHON_RUN" scripts/encode.py \
      --input "$VALID_TEXT" \
      --output "$VALID_BIN" \
      --vocab "$VOCAB_PATH" \
      --merges "$MERGES_PATH" \
      --special-token '<|endoftext|>'
    log "valid encoding done: $VALID_BIN"
  else
    log "skip valid encoding: existing $VALID_BIN"
  fi
}

write_config() {
  mkdir -p runs/configs
  cat > "$CONFIG_PATH" <<EOF
RUN_NAME='$RUN_NAME'
TOKENIZER_RUN_NAME='$TOKENIZER_RUN_NAME'
TRAIN_TEXT='$TRAIN_TEXT'
VALID_TEXT='$VALID_TEXT'
TOKENIZER_TRAIN_TEXT='$TOKENIZER_TRAIN_TEXT'
VOCAB_SIZE='$VOCAB_SIZE'
CONTEXT_LENGTH='$CONTEXT_LENGTH'
D_MODEL='$D_MODEL'
NUM_LAYERS='$NUM_LAYERS'
NUM_HEADS='$NUM_HEADS'
D_FF='$D_FF'
ROPE_THETA='$ROPE_THETA'
BATCH_SIZE='$BATCH_SIZE'
STEPS='$STEPS'
LR='$LR'
MIN_LR='$MIN_LR'
WARMUP_STEPS='$WARMUP_STEPS'
GRAD_CLIP='$GRAD_CLIP'
LOG_EVERY='$LOG_EVERY'
EVAL_EVERY='$EVAL_EVERY'
EVAL_BATCHES='$EVAL_BATCHES'
CHECKPOINT_EVERY='$CHECKPOINT_EVERY'
SEED='$SEED'
DEVICE='$DEVICE'
PYTHON_RUN='$PYTHON_RUN'
TF32='$TF32'
COMPILE='$COMPILE'
VOCAB_PATH='$VOCAB_PATH'
MERGES_PATH='$MERGES_PATH'
TRAIN_BIN='$TRAIN_BIN'
VALID_BIN='$VALID_BIN'
CHECKPOINT='$CHECKPOINT'
TRAIN_LOG='$TRAIN_LOG'
EOF
}

train_lm() {
  if [[ "$FORCE_TRAIN" != "1" && -f "$CHECKPOINT" ]]; then
    log "skip training: existing checkpoint $CHECKPOINT"
    return 0
  fi

  local resume_path
  resume_path=""
  if [[ "$RESUME" == "1" ]]; then
    if [[ -f "$CHECKPOINT" ]]; then
      resume_path="$CHECKPOINT"
    else
      resume_path="$(latest_step_checkpoint "$CHECKPOINT")"
    fi
  fi

  local thread_args tf32_args compile_args resume_args
  thread_args=()
  tf32_args=()
  compile_args=()
  resume_args=()
  if [[ -n "$NUM_THREADS" ]]; then
    thread_args=(--num-threads "$NUM_THREADS")
  fi
  if [[ "$TF32" == "1" ]]; then
    tf32_args=(--tf32)
  fi
  if [[ "$COMPILE" == "1" ]]; then
    compile_args=(--compile)
  fi
  if [[ -n "$resume_path" ]]; then
    resume_args=(--resume "$resume_path")
  fi

  local cmd=(
    "$PYTHON_RUN" scripts/train_lm.py
    --train-data "$TRAIN_BIN"
    --valid-data "$VALID_BIN"
    --output "$CHECKPOINT"
    --dtype uint16
    --vocab-size "$VOCAB_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --d-model "$D_MODEL"
    --num-layers "$NUM_LAYERS"
    --num-heads "$NUM_HEADS"
    --d-ff "$D_FF"
    --rope-theta "$ROPE_THETA"
    --batch-size "$BATCH_SIZE"
    --steps "$STEPS"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --warmup-steps "$WARMUP_STEPS"
    --grad-clip "$GRAD_CLIP"
    --log-every "$LOG_EVERY"
    --eval-every "$EVAL_EVERY"
    --eval-batches "$EVAL_BATCHES"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --log-file "$TRAIN_LOG"
    --seed "$SEED"
    --device "$DEVICE"
    "${thread_args[@]}"
    "${tf32_args[@]}"
    "${compile_args[@]}"
    "${resume_args[@]}"
  )

  log "train LM: checkpoint=$CHECKPOINT log=$TRAIN_LOG"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$CONSOLE_LOG"
  log "done training: $RUN_NAME"
}

main() {
  mkdir -p runs/tokenizers runs/encoded runs/checkpoints runs/logs runs/configs
  check_runtime
  train_tokenizer_if_needed
  encode_if_needed
  write_config
  train_lm
}

main "$@"
