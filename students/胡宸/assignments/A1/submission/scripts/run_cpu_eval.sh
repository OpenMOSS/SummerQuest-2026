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
PROMPT="${PROMPT:-Once upon a time}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
CONFIG_PATH="${CONFIG_PATH:-runs/configs/${RUN_NAME}.env}"

if [[ -f "$CONFIG_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_PATH"
fi

case "$PROFILE" in
  tiny)
    VOCAB_SIZE="${VOCAB_SIZE:-1000}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-64}"
    D_MODEL="${D_MODEL:-64}"
    NUM_LAYERS="${NUM_LAYERS:-2}"
    NUM_HEADS="${NUM_HEADS:-4}"
    D_FF="${D_FF:-192}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    EVAL_BATCHES="${EVAL_BATCHES:-8}"
    ;;
  small)
    VOCAB_SIZE="${VOCAB_SIZE:-5000}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-128}"
    D_MODEL="${D_MODEL:-128}"
    NUM_LAYERS="${NUM_LAYERS:-2}"
    NUM_HEADS="${NUM_HEADS:-4}"
    D_FF="${D_FF:-384}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    EVAL_BATCHES="${EVAL_BATCHES:-16}"
    ;;
  main)
    VOCAB_SIZE="${VOCAB_SIZE:-10000}"
    CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
    D_MODEL="${D_MODEL:-512}"
    NUM_LAYERS="${NUM_LAYERS:-4}"
    NUM_HEADS="${NUM_HEADS:-16}"
    D_FF="${D_FF:-1344}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    EVAL_BATCHES="${EVAL_BATCHES:-32}"
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE. Use tiny, small, or main." >&2
    exit 2
    ;;
esac

VOCAB_PATH="${VOCAB_PATH:-runs/tokenizers/${RUN_NAME}_vocab.json}"
MERGES_PATH="${MERGES_PATH:-runs/tokenizers/${RUN_NAME}_merges.txt}"
VALID_BIN="${VALID_BIN:-runs/encoded/${RUN_NAME}_valid.bin}"
CHECKPOINT="${CHECKPOINT:-runs/checkpoints/${RUN_NAME}.pt}"
EVAL_LOG="${EVAL_LOG:-runs/logs/${RUN_NAME}_eval.json}"

for path in "$VOCAB_PATH" "$MERGES_PATH" "$VALID_BIN" "$CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    echo "Run training first: PROFILE=$PROFILE RUN_NAME=$RUN_NAME scripts/run_cpu_training.sh" >&2
    exit 1
  fi
done

THREAD_ARGS=()
if [[ -n "$NUM_THREADS" ]]; then
  THREAD_ARGS=(--num-threads "$NUM_THREADS")
fi

$PYTHON_RUN scripts/evaluate_lm.py \
  --checkpoint "$CHECKPOINT" \
  --eval-data "$VALID_BIN" \
  --dtype uint16 \
  --vocab "$VOCAB_PATH" \
  --merges "$MERGES_PATH" \
  --vocab-size "$VOCAB_SIZE" \
  --context-length "$CONTEXT_LENGTH" \
  --d-model "$D_MODEL" \
  --num-layers "$NUM_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --d-ff "$D_FF" \
  --batch-size "$BATCH_SIZE" \
  --eval-batches "$EVAL_BATCHES" \
  --prompt "$PROMPT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-file "$EVAL_LOG" \
  --device "$DEVICE" \
  "${THREAD_ARGS[@]}"

echo "Evaluation log: $EVAL_LOG"
