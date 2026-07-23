#!/usr/bin/env bash
set -euo pipefail

if command -v python >/dev/null 2>&1; then PYTHON=(python); else PYTHON=(uv run python); fi

OUT_DIR="${OUT_DIR:-logs/ablations_full_val}"
CKPT_ROOT="${CKPT_ROOT:-checkpoint/ablations}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-auto}"
APPEND_LOG="${APPEND_LOG:-$OUT_DIR/summary.jsonl}"

mkdir -p "$OUT_DIR"
: > "$APPEND_LOG"

run_eval() {
  local name="$1"
  local checkpoint="$CKPT_ROOT/$name/best.pt"
  shift

  "${PYTHON[@]}" scripts/evaluate_checkpoint.py \
    --config configs/train_tinystories.json \
    --checkpoint "$checkpoint" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output "$OUT_DIR/$name.json" \
    --append-log "$APPEND_LOG" \
    --set "run.name=$name" \
    "$@"
}

run_eval remove_rmsnorm --set model.norm_mode=none
run_eval post_norm --set model.norm_mode=post
run_eval nope --set model.use_rope=false
run_eval silu_ffn --set model.ffn_type=silu --set model.d_ff=2048
