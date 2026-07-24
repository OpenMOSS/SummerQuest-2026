#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"
PROMPT="${PROMPT:-Once upon a time}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

if [[ ! -d ".venv" ]]; then
  uv sync --frozen --no-install-project
fi

UV_RUN="uv run --no-sync"

TS_TOK_DIR="${TS_TOK_DIR:-artifacts/tokenizer_tinystories}"
OWT_TOK_DIR="${OWT_TOK_DIR:-artifacts/tokenizer_owt}"

TS_TRAIN_TXT="${TS_TRAIN_TXT:-data/TinyStoriesV2-GPT4-train.txt}"
TS_VALID_TXT="${TS_VALID_TXT:-data/TinyStoriesV2-GPT4-valid.txt}"
OWT_TRAIN_TXT="${OWT_TRAIN_TXT:-data/owt_train.txt}"
OWT_VALID_TXT="${OWT_VALID_TXT:-data/owt_valid.txt}"

TS_TRAIN_NPY="${TS_TRAIN_NPY:-artifacts/datasets/tinystories_train.npy}"
TS_VALID_NPY="${TS_VALID_NPY:-artifacts/datasets/tinystories_valid.npy}"
OWT_TRAIN_NPY="${OWT_TRAIN_NPY:-artifacts/datasets/owt_train.npy}"
OWT_VALID_NPY="${OWT_VALID_NPY:-artifacts/datasets/owt_valid.npy}"

TS_RUN_DIR="${TS_RUN_DIR:-artifacts/runs/tinystories_baseline}"
OWT_RUN_DIR="${OWT_RUN_DIR:-artifacts/runs/owt_baseline}"

TS_TRAIN_CFG="${TS_TRAIN_CFG:-configs/train_tinystories.json}"
OWT_TRAIN_CFG="${OWT_TRAIN_CFG:-configs/train_owt.json}"
TS_TOK_CFG="${TS_TOK_CFG:-configs/tokenizer_tinystories.json}"
OWT_TOK_CFG="${OWT_TOK_CFG:-configs/tokenizer_owt.json}"

mkdir -p artifacts/datasets artifacts/runs logs logs/lr_sweep logs/batch_size logs/ablations

run_config_group() {
  local group_name="$1"
  local config_dir="$2"
  local out_root="$3"
  local log_dir="$4"
  local -a configs=()
  local cfg=""
  local total=0
  local index=0

  shopt -s nullglob
  configs=("$config_dir"/*.json)
  shopt -u nullglob

  total="${#configs[@]}"
  if [[ "$total" -eq 0 ]]; then
    echo "No configs found under $config_dir/"
    exit 2
  fi

  echo "[$group_name] found $total configs"
  for cfg in "${configs[@]}"; do
    local name out_dir start_ts end_ts elapsed
    index=$((index + 1))
    name="$(basename "$cfg" .json)"
    out_dir="${out_root}/${name}"
    start_ts="$(date +%s)"
    echo "[$group_name][$index/$total] start: ${name} (config=${cfg})"
    $UV_RUN -m scripts.train --config "$cfg" --device "$DEVICE"
    if [[ -f "$out_dir/train.jsonl" ]]; then
      cp "$out_dir/train.jsonl" "${log_dir}/${name}.jsonl"
    fi
    end_ts="$(date +%s)"
    elapsed=$((end_ts - start_ts))
    echo "[$group_name][$index/$total] done: ${name} (${elapsed}s)"
  done
}

usage() {
  cat <<'USAGE'
Usage:
  ./run_a1.sh <command>

Commands:
  test
  tokenizer_tinystories
  tokenizer_owt
  encode_tinystories
  encode_owt
  train_tinystories
  train_owt
  generate_tinystories
  generate_owt
  tinystories_end_to_end
  owt_end_to_end
  lr_sweep
  batch_sweep
  ablations

Environment variables (optional):
  DEVICE=auto|cuda|cpu
  PROMPT=...
  MAX_NEW_TOKENS=256
  TEMPERATURE=0.8
  TOP_P=0.95
USAGE
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 2
fi
shift || true

case "$cmd" in
  test)
    $UV_RUN pytest
    ;;

  tokenizer_tinystories)
    $UV_RUN -m scripts.train_tokenizer --config "$TS_TOK_CFG" --output_dir "$TS_TOK_DIR"
    ;;

  tokenizer_owt)
    $UV_RUN -m scripts.train_tokenizer --config "$OWT_TOK_CFG" --output_dir "$OWT_TOK_DIR"
    ;;

  encode_tinystories)
    $UV_RUN -m scripts.encode_corpus \
      --config "$TS_TOK_CFG" \
      --tokenizer_dir "$TS_TOK_DIR" \
      --input_path "$TS_TRAIN_TXT" \
      --output_path "$TS_TRAIN_NPY"

    $UV_RUN -m scripts.encode_corpus \
      --config "$TS_TOK_CFG" \
      --tokenizer_dir "$TS_TOK_DIR" \
      --input_path "$TS_VALID_TXT" \
      --output_path "$TS_VALID_NPY"
    ;;

  encode_owt)
    $UV_RUN -m scripts.encode_corpus \
      --config "$OWT_TOK_CFG" \
      --tokenizer_dir "$OWT_TOK_DIR" \
      --input_path "$OWT_TRAIN_TXT" \
      --output_path "$OWT_TRAIN_NPY"

    $UV_RUN -m scripts.encode_corpus \
      --config "$OWT_TOK_CFG" \
      --tokenizer_dir "$OWT_TOK_DIR" \
      --input_path "$OWT_VALID_TXT" \
      --output_path "$OWT_VALID_NPY"
    ;;

  train_tinystories)
    $UV_RUN -m scripts.train --config "$TS_TRAIN_CFG" --device "$DEVICE"
    if [[ -f "$TS_RUN_DIR/train.jsonl" ]]; then
      cp "$TS_RUN_DIR/train.jsonl" "logs/train_tinystories.jsonl"
    fi
    ;;

  train_owt)
    $UV_RUN -m scripts.train --config "$OWT_TRAIN_CFG" --device "$DEVICE"
    if [[ -f "$OWT_RUN_DIR/train.jsonl" ]]; then
      cp "$OWT_RUN_DIR/train.jsonl" "logs/train_owt.jsonl"
    fi
    ;;

  generate_tinystories)
    $UV_RUN -m scripts.generate \
      --tokenizer_dir "$TS_TOK_DIR" \
      --model_config "$TS_RUN_DIR/model_config.json" \
      --checkpoint "$TS_RUN_DIR/checkpoint_final.pt" \
      --prompt "$PROMPT" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --device "$DEVICE"
    ;;

  generate_owt)
    $UV_RUN -m scripts.generate \
      --tokenizer_dir "$OWT_TOK_DIR" \
      --model_config "$OWT_RUN_DIR/model_config.json" \
      --checkpoint "$OWT_RUN_DIR/checkpoint_final.pt" \
      --prompt "$PROMPT" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top_p "$TOP_P" \
      --device "$DEVICE"
    ;;

  tinystories_end_to_end)
    "$0" tokenizer_tinystories
    "$0" encode_tinystories
    "$0" train_tinystories
    "$0" generate_tinystories
    ;;

  owt_end_to_end)
    "$0" tokenizer_owt
    "$0" encode_owt
    "$0" train_owt
    "$0" generate_owt
    ;;

  lr_sweep)
    run_config_group "lr_sweep" "configs/lr_sweep" "artifacts/runs/lr_sweep" "logs/lr_sweep"
    ;;

  batch_sweep)
    run_config_group "batch_sweep" "configs/batch_size" "artifacts/runs/batch_size" "logs/batch_size"
    ;;

  ablations)
    run_config_group "ablations" "configs/ablations" "artifacts/runs/ablations" "logs/ablations"
    ;;

  *)
    usage
    exit 2
    ;;
esac
