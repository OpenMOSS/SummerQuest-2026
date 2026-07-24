#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

TARGET="${TARGET:-all}"
NUM_THREADS="${NUM_THREADS:-8}"
PYTHON_RUN="${PYTHON_RUN:-uv run python}"
RUN_EVAL="${RUN_EVAL:-1}"
FULL_TOKENIZER="${FULL_TOKENIZER:-0}"

mkdir -p runs/logs

run_one() {
  local dataset="$1"
  local run_name="$2"
  shift 2

  echo "===== training $run_name ====="
  env \
    DATASET="$dataset" \
    PROFILE=main \
    RUN_NAME="$run_name" \
    NUM_THREADS="$NUM_THREADS" \
    PYTHON_RUN="$PYTHON_RUN" \
    "$@" \
    scripts/run_cpu_training.sh 2>&1 | tee "runs/logs/${run_name}.console.log"

  if [[ "$RUN_EVAL" == "1" ]]; then
    echo "===== evaluating $run_name ====="
    env \
      DATASET="$dataset" \
      PROFILE=main \
      RUN_NAME="$run_name" \
      NUM_THREADS="$NUM_THREADS" \
      PYTHON_RUN="$PYTHON_RUN" \
      scripts/run_cpu_eval.sh 2>&1 | tee "runs/logs/${run_name}.eval.console.log"
  fi
}

case "$TARGET" in
  tinystories)
    if [[ "$FULL_TOKENIZER" == "1" ]]; then
      run_one tinystories tinystories_main_cpu_full_tokenizer
    else
      run_one tinystories tinystories_main_cpu TOKENIZER_TRAIN_TEXT=data/TinyStoriesV2-GPT4-valid.txt
    fi
    ;;
  owt)
    if [[ "$FULL_TOKENIZER" == "1" || "${FULL_OWT_TOKENIZER:-0}" == "1" ]]; then
      run_one owt owt_main_cpu_full_tokenizer ALLOW_HUGE_TOKENIZER=1
    else
      run_one owt owt_main_cpu TOKENIZER_TRAIN_TEXT=data/owt_valid.txt
    fi
    ;;
  all)
    if [[ "$FULL_TOKENIZER" == "1" ]]; then
      run_one tinystories tinystories_main_cpu_full_tokenizer
    else
      run_one tinystories tinystories_main_cpu TOKENIZER_TRAIN_TEXT=data/TinyStoriesV2-GPT4-valid.txt
    fi
    if [[ "$FULL_TOKENIZER" == "1" || "${FULL_OWT_TOKENIZER:-0}" == "1" ]]; then
      run_one owt owt_main_cpu_full_tokenizer ALLOW_HUGE_TOKENIZER=1
    else
      run_one owt owt_main_cpu TOKENIZER_TRAIN_TEXT=data/owt_valid.txt
    fi
    ;;
  *)
    echo "Unknown TARGET=$TARGET. Use tinystories, owt, or all." >&2
    exit 2
    ;;
esac
