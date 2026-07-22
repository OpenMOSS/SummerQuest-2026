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
RUN_ROOT="${RUN_ROOT:-runs}"
RUN_DIR="${RUN_DIR:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-owt_baseline}"
RESUME="${RESUME:-}"
OVERWRITE="${OVERWRITE:-0}"
NO_AUTO_NAME="${NO_AUTO_NAME:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

VOCAB_SIZE="${VOCAB_SIZE:-32000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
D_FF="${D_FF:-1344}"
STEPS="${STEPS:-5000}"
EVAL_ITERS="${EVAL_ITERS:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1000}"
MAX_LR="${MAX_LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_ITERS="${WARMUP_ITERS:-1000}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
SEED="${SEED:-42}"
NUM_THREADS="${NUM_THREADS:-40}"
PEAK_FLOPS="${PEAK_FLOPS:-989e12}"
NORM_TYPE="${NORM_TYPE:-rmsnorm}"
NORM_POSITION="${NORM_POSITION:-pre}"
POS_EMB="${POS_EMB:-rope}"
FFN_TYPE="${FFN_TYPE:-swiglu}"

ARGS=(
  scripts/train_lm.py
  --train-data "$DATA_DIR/owt_train.npy"
  --val-data "$DATA_DIR/owt_val.npy"
  --run-root "$RUN_ROOT"
  --experiment-name "$EXPERIMENT_NAME"
  --vocab-size "$VOCAB_SIZE"
  --context-length 256
  --d-model 512
  --d-ff "$D_FF"
  --num-layers 4
  --num-heads 16
  --rope-theta 10000
  --norm-type "$NORM_TYPE"
  --norm-position "$NORM_POSITION"
  --pos-emb "$POS_EMB"
  --ffn-type "$FFN_TYPE"
  --batch-size "$BATCH_SIZE"
  --steps "$STEPS"
  --eval-iters "$EVAL_ITERS"
  --eval-interval "$EVAL_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --max-lr "$MAX_LR"
  --min-lr "$MIN_LR"
  --warmup-iters "$WARMUP_ITERS"
  --weight-decay "$WEIGHT_DECAY"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --seed "$SEED"
  --num-threads "$NUM_THREADS"
  --peak-flops "$PEAK_FLOPS"
)

if [[ -n "$RUN_DIR" ]]; then
  ARGS+=(--out-dir "$RUN_DIR")
fi
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "$NO_AUTO_NAME" == "1" ]]; then
  ARGS+=(--no-auto-name)
fi

"$PYTHON" "${ARGS[@]}"
