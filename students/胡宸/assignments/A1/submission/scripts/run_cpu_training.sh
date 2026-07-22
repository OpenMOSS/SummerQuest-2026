#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

PROFILE="${PROFILE:-tiny}"
DATASET="${DATASET:-tinystories}"
RUN_NAME="${RUN_NAME:-${DATASET}_${PROFILE}_cpu}"
PYTHON_RUN="${PYTHON_RUN:-uv run python}"
DEVICE="${DEVICE:-cpu}"
NUM_THREADS="${NUM_THREADS:-}"
FORCE="${FORCE:-0}"
ALLOW_HUGE_TOKENIZER="${ALLOW_HUGE_TOKENIZER:-0}"
MAX_TOKENIZER_BYTES="${MAX_TOKENIZER_BYTES:-}"

log_stage() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >&2
}

case "$PROFILE" in
  tiny)
    TRAIN_TEXT="${TRAIN_TEXT:-tests/fixtures/tinystories_sample_5M.txt}"
    VALID_TEXT="${VALID_TEXT:-tests/fixtures/tinystories_sample.txt}"
    VOCAB_SIZE="${VOCAB_SIZE:-1000}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
    D_MODEL="${D_MODEL:-64}"
    NUM_LAYERS="${NUM_LAYERS:-2}"
    NUM_HEADS="${NUM_HEADS:-4}"
    D_FF="${D_FF:-192}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    STEPS="${STEPS:-50}"
    LR="${LR:-3e-4}"
    MIN_LR="${MIN_LR:-3e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-5}"
    LOG_EVERY="${LOG_EVERY:-5}"
    EVAL_EVERY="${EVAL_EVERY:-10}"
    EVAL_BATCHES="${EVAL_BATCHES:-4}"
    ;;
  small)
    if [[ "$DATASET" == "owt" ]]; then
      TRAIN_TEXT="${TRAIN_TEXT:-data/owt_train.txt}"
      VALID_TEXT="${VALID_TEXT:-data/owt_valid.txt}"
      VOCAB_SIZE="${VOCAB_SIZE:-8000}"
    elif [[ "$DATASET" == "tinystories" ]]; then
      TRAIN_TEXT="${TRAIN_TEXT:-data/TinyStoriesV2-GPT4-train.txt}"
      VALID_TEXT="${VALID_TEXT:-data/TinyStoriesV2-GPT4-valid.txt}"
      VOCAB_SIZE="${VOCAB_SIZE:-5000}"
    else
      echo "Unknown DATASET=$DATASET. Use tinystories or owt." >&2
      exit 2
    fi
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
    D_MODEL="${D_MODEL:-128}"
    NUM_LAYERS="${NUM_LAYERS:-2}"
    NUM_HEADS="${NUM_HEADS:-4}"
    D_FF="${D_FF:-384}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    STEPS="${STEPS:-1000}"
    LR="${LR:-3e-4}"
    MIN_LR="${MIN_LR:-3e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-100}"
    LOG_EVERY="${LOG_EVERY:-20}"
    EVAL_EVERY="${EVAL_EVERY:-100}"
    EVAL_BATCHES="${EVAL_BATCHES:-8}"
    ;;
  main)
    if [[ "$DATASET" == "owt" ]]; then
      TRAIN_TEXT="${TRAIN_TEXT:-data/owt_train.txt}"
      VALID_TEXT="${VALID_TEXT:-data/owt_valid.txt}"
      VOCAB_SIZE="${VOCAB_SIZE:-32000}"
    elif [[ "$DATASET" == "tinystories" ]]; then
      TRAIN_TEXT="${TRAIN_TEXT:-data/TinyStoriesV2-GPT4-train.txt}"
      VALID_TEXT="${VALID_TEXT:-data/TinyStoriesV2-GPT4-valid.txt}"
      VOCAB_SIZE="${VOCAB_SIZE:-10000}"
    else
      echo "Unknown DATASET=$DATASET. Use tinystories or owt." >&2
      exit 2
    fi
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
    D_MODEL="${D_MODEL:-512}"
    NUM_LAYERS="${NUM_LAYERS:-4}"
    NUM_HEADS="${NUM_HEADS:-16}"
    D_FF="${D_FF:-1344}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    STEPS="${STEPS:-10000}"
    LR="${LR:-3e-4}"
    MIN_LR="${MIN_LR:-3e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-100}"
    LOG_EVERY="${LOG_EVERY:-50}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    EVAL_BATCHES="${EVAL_BATCHES:-16}"
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE. Use tiny, small, or main." >&2
    exit 2
    ;;
esac

if [[ -z "$MAX_TOKENIZER_BYTES" ]]; then
  if [[ "$DATASET" == "owt" ]]; then
    MAX_TOKENIZER_BYTES=1073741824
  else
    MAX_TOKENIZER_BYTES=4294967296
  fi
fi

if [[ ! -f "$TRAIN_TEXT" || ! -f "$VALID_TEXT" ]]; then
  cat >&2 <<EOF
Missing training data.
TRAIN_TEXT=$TRAIN_TEXT
VALID_TEXT=$VALID_TEXT

For PROFILE=small/main, download data first:
  mkdir -p data && cd data
  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
  wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz && gunzip owt_train.txt.gz
  wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz && gunzip owt_valid.txt.gz
EOF
  exit 1
fi

TOKENIZER_TRAIN_TEXT="${TOKENIZER_TRAIN_TEXT:-$TRAIN_TEXT}"
if [[ ! -f "$TOKENIZER_TRAIN_TEXT" ]]; then
  echo "Missing tokenizer training text: $TOKENIZER_TRAIN_TEXT" >&2
  exit 1
fi

mkdir -p runs/tokenizers runs/encoded runs/checkpoints runs/logs runs/configs

VOCAB_PATH="runs/tokenizers/${RUN_NAME}_vocab.json"
MERGES_PATH="runs/tokenizers/${RUN_NAME}_merges.txt"
TRAIN_BIN="runs/encoded/${RUN_NAME}_train.bin"
VALID_BIN="runs/encoded/${RUN_NAME}_valid.bin"
CHECKPOINT="runs/checkpoints/${RUN_NAME}.pt"
TRAIN_LOG="runs/logs/${RUN_NAME}_train.jsonl"
CONFIG_PATH="runs/configs/${RUN_NAME}.env"

cat > "$CONFIG_PATH" <<EOF
PROFILE='$PROFILE'
DATASET='$DATASET'
RUN_NAME='$RUN_NAME'
TOKENIZER_TRAIN_TEXT='$TOKENIZER_TRAIN_TEXT'
VOCAB_SIZE='$VOCAB_SIZE'
CONTEXT_LENGTH='$CONTEXT_LENGTH'
D_MODEL='$D_MODEL'
NUM_LAYERS='$NUM_LAYERS'
NUM_HEADS='$NUM_HEADS'
D_FF='$D_FF'
BATCH_SIZE='$BATCH_SIZE'
VOCAB_PATH='$VOCAB_PATH'
MERGES_PATH='$MERGES_PATH'
VALID_BIN='$VALID_BIN'
CHECKPOINT='$CHECKPOINT'
EOF

if [[ "$FORCE" == "1" || ! -f "$VOCAB_PATH" || ! -f "$MERGES_PATH" ]]; then
  TOKENIZER_BYTES="$(wc -c < "$TOKENIZER_TRAIN_TEXT")"
  if [[ "$TOKENIZER_BYTES" -gt "$MAX_TOKENIZER_BYTES" && "$ALLOW_HUGE_TOKENIZER" != "1" ]]; then
    cat >&2 <<EOF
Refusing to train BPE tokenizer on a very large text file with this from-scratch Python implementation.

TOKENIZER_TRAIN_TEXT=$TOKENIZER_TRAIN_TEXT
size_bytes=$TOKENIZER_BYTES
MAX_TOKENIZER_BYTES=$MAX_TOKENIZER_BYTES

The LM can still train on the full encoded dataset, but tokenizer training should use a smaller text
such as data/owt_valid.txt or a prepared subset. For example:

  DATASET=owt PROFILE=main RUN_NAME=$RUN_NAME \\
    TOKENIZER_TRAIN_TEXT=data/owt_valid.txt NUM_THREADS=8 \\
    scripts/run_cpu_training.sh

To force full tokenizer training anyway, set ALLOW_HUGE_TOKENIZER=1. It may be killed by the OS.
EOF
    exit 1
  fi
  log_stage "train tokenizer: input=$TOKENIZER_TRAIN_TEXT vocab_size=$VOCAB_SIZE"
  $PYTHON_RUN scripts/train_tokenizer.py \
    --input "$TOKENIZER_TRAIN_TEXT" \
    --vocab-size "$VOCAB_SIZE" \
    --vocab-out "$VOCAB_PATH" \
    --merges-out "$MERGES_PATH"
  log_stage "tokenizer done: vocab=$VOCAB_PATH merges=$MERGES_PATH"
else
  log_stage "skip tokenizer: existing vocab=$VOCAB_PATH merges=$MERGES_PATH"
fi

if [[ "$FORCE" == "1" || ! -f "$TRAIN_BIN" ]]; then
  log_stage "encode train split: input=$TRAIN_TEXT output=$TRAIN_BIN"
  $PYTHON_RUN scripts/encode.py \
    --input "$TRAIN_TEXT" \
    --output "$TRAIN_BIN" \
    --vocab "$VOCAB_PATH" \
    --merges "$MERGES_PATH" \
    --special-token '<|endoftext|>'
  log_stage "train encoding done: $TRAIN_BIN"
else
  log_stage "skip train encoding: existing $TRAIN_BIN"
fi

if [[ "$FORCE" == "1" || ! -f "$VALID_BIN" ]]; then
  log_stage "encode valid split: input=$VALID_TEXT output=$VALID_BIN"
  $PYTHON_RUN scripts/encode.py \
    --input "$VALID_TEXT" \
    --output "$VALID_BIN" \
    --vocab "$VOCAB_PATH" \
    --merges "$MERGES_PATH" \
    --special-token '<|endoftext|>'
  log_stage "valid encoding done: $VALID_BIN"
else
  log_stage "skip valid encoding: existing $VALID_BIN"
fi

THREAD_ARGS=()
if [[ -n "$NUM_THREADS" ]]; then
  THREAD_ARGS=(--num-threads "$NUM_THREADS")
fi

log_stage "train LM: checkpoint=$CHECKPOINT log=$TRAIN_LOG"
$PYTHON_RUN scripts/train_lm.py \
  --train-data "$TRAIN_BIN" \
  --valid-data "$VALID_BIN" \
  --output "$CHECKPOINT" \
  --dtype uint16 \
  --vocab-size "$VOCAB_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --d-ff "$D_FF" \
  --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --lr "$LR" \
  --min-lr "$MIN_LR" \
  --warmup-steps "$WARMUP_STEPS" \
  --log-every "$LOG_EVERY" \
  --eval-every "$EVAL_EVERY" \
  --eval-batches "$EVAL_BATCHES" \
  --log-file "$TRAIN_LOG" \
  --device "$DEVICE" \
  "${THREAD_ARGS[@]}"

cat <<EOF
Training finished.
checkpoint: $CHECKPOINT
train log:  $TRAIN_LOG
run config: $CONFIG_PATH
vocab:      $VOCAB_PATH
merges:     $MERGES_PATH
valid bin:  $VALID_BIN

Run evaluation:
  PROFILE=$PROFILE RUN_NAME=$RUN_NAME scripts/run_cpu_eval.sh
EOF
