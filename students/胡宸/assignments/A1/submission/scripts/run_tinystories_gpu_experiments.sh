#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_RUN="${PYTHON_RUN:-.venv/bin/python}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-.tiktoken-cache}"

RUN_PREFIX="${RUN_PREFIX:-tinystories_gpu}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_LR_SWEEP="${RUN_LR_SWEEP:-1}"
RUN_BATCH_SWEEP="${RUN_BATCH_SWEEP:-1}"
RUN_GENERATION="${RUN_GENERATION:-1}"
SKIP_UNSUPPORTED_ABLATIONS="${SKIP_UNSUPPORTED_ABLATIONS:-0}"

TRAIN_DATA="${TRAIN_DATA:-runs/encoded/tinystories_main_cpu_full_tokenizer_train.bin}"
VALID_DATA="${VALID_DATA:-runs/encoded/tinystories_main_cpu_full_tokenizer_valid.bin}"
VOCAB_PATH="${VOCAB_PATH:-runs/tokenizers/tinystories_main_cpu_full_tokenizer_vocab.json}"
MERGES_PATH="${MERGES_PATH:-runs/tokenizers/tinystories_main_cpu_full_tokenizer_merges.txt}"

VOCAB_SIZE="${VOCAB_SIZE:-10000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-16}"
D_FF="${D_FF:-1344}"
ROPE_THETA="${ROPE_THETA:-10000.0}"

BATCH_SIZE="${BATCH_SIZE:-128}"
STEPS="${STEPS:-10000}"
ABLATION_STEPS="${ABLATION_STEPS:-$STEPS}"
LR_SWEEP_STEPS="${LR_SWEEP_STEPS:-2000}"
BATCH_SWEEP_STEPS="${BATCH_SWEEP_STEPS:-10000}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
SEED="${SEED:-0}"
NUM_THREADS="${NUM_THREADS:-}"
TF32="${TF32:-1}"
COMPILE="${COMPILE:-0}"

LR_SWEEP_VALUES="${LR_SWEEP_VALUES:-1e-4 3e-4 1e-3 3e-3 1e-1}"
BATCH_SWEEP_VALUES="${BATCH_SWEEP_VALUES:-1 2 4 8 16 32 64 128}"
GENERATE_PROMPT="${GENERATE_PROMPT:-Once upon a time}"
GENERATE_MAX_NEW_TOKENS="${GENERATE_MAX_NEW_TOKENS:-256}"
GENERATE_TEMPERATURE="${GENERATE_TEMPERATURE:-0.8}"
GENERATE_TOP_P="${GENERATE_TOP_P:-0.9}"
ALLOW_EXPERIMENT_FAILURES="${ALLOW_EXPERIMENT_FAILURES:-0}"

mkdir -p runs/checkpoints runs/logs runs/experiment_manifests

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

sanitize_label() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  value="${value//+/p}"
  printf '%s\n' "$value"
}

require_flag_support() {
  local label="$1"
  shift
  local help_text
  help_text="$("$PYTHON_RUN" scripts/train_lm.py --help 2>&1 || true)"
  local flag
  for flag in "$@"; do
    if [[ "$help_text" != *"$flag"* ]]; then
      if [[ "$SKIP_UNSUPPORTED_ABLATIONS" == "1" ]]; then
        log "skip $label: scripts/train_lm.py does not support $flag"
        return 1
      fi
      cat >&2 <<EOF
Unsupported ablation flag for $label: $flag

Current scripts/train_lm.py does not expose this ablation switch. Add the
corresponding model/training CLI support first, or rerun with:

  SKIP_UNSUPPORTED_ABLATIONS=1 $0

This guard prevents accidentally running every ablation as the baseline model.
EOF
      exit 2
    fi
  done
  return 0
}

check_runtime() {
  require_file "$PYTHON_RUN"
  require_file "$TRAIN_DATA"
  require_file "$VALID_DATA"
  if [[ "$RUN_GENERATION" == "1" ]]; then
    require_file "$VOCAB_PATH"
    require_file "$MERGES_PATH"
  fi

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

run_train() {
  local label="$1"
  local steps="$2"
  shift 2
  local extra_args=("$@")

  local run_name output log_file console_log manifest resume_path
  run_name="${RUN_PREFIX}_${label}"
  output="runs/checkpoints/${run_name}.pt"
  log_file="runs/logs/${run_name}_train.jsonl"
  console_log="runs/logs/${run_name}.console.log"
  manifest="runs/experiment_manifests/${run_name}.env"

  if [[ "$FORCE" != "1" && -f "$output" ]]; then
    log "skip $run_name: existing checkpoint $output"
    return 0
  fi

  resume_path=""
  if [[ "$RESUME" == "1" && ! -f "$output" ]]; then
    resume_path="$(latest_step_checkpoint "$output")"
  fi

  cat > "$manifest" <<EOF
RUN_NAME='$run_name'
LABEL='$label'
TRAIN_DATA='$TRAIN_DATA'
VALID_DATA='$VALID_DATA'
VOCAB_SIZE='$VOCAB_SIZE'
CONTEXT_LENGTH='$CONTEXT_LENGTH'
D_MODEL='$D_MODEL'
NUM_LAYERS='$NUM_LAYERS'
NUM_HEADS='$NUM_HEADS'
D_FF='$D_FF'
ROPE_THETA='$ROPE_THETA'
BATCH_SIZE='$BATCH_SIZE'
STEPS='$steps'
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
EXTRA_ARGS='${extra_args[*]}'
RESUME_PATH='$resume_path'
EOF

  local cmd=(
    "$PYTHON_RUN" scripts/train_lm.py
    --train-data "$TRAIN_DATA"
    --valid-data "$VALID_DATA"
    --output "$output"
    --dtype uint16
    --vocab-size "$VOCAB_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --d-model "$D_MODEL"
    --num-layers "$NUM_LAYERS"
    --num-heads "$NUM_HEADS"
    --d-ff "$D_FF"
    --rope-theta "$ROPE_THETA"
    --batch-size "$BATCH_SIZE"
    --steps "$steps"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --warmup-steps "$WARMUP_STEPS"
    --grad-clip "$GRAD_CLIP"
    --log-every "$LOG_EVERY"
    --eval-every "$EVAL_EVERY"
    --eval-batches "$EVAL_BATCHES"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --log-file "$log_file"
    --seed "$SEED"
    --device "$DEVICE"
  )

  if [[ "$TF32" == "1" ]]; then
    cmd+=(--tf32)
  fi
  if [[ "$COMPILE" == "1" ]]; then
    cmd+=(--compile)
  fi
  if [[ -n "$NUM_THREADS" ]]; then
    cmd+=(--num-threads "$NUM_THREADS")
  fi
  if [[ -n "$resume_path" ]]; then
    cmd+=(--resume "$resume_path")
  fi
  cmd+=("${extra_args[@]}")

  log "start $run_name"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$console_log"
  log "done $run_name"
}

run_train_allowing_failure() {
  if [[ "$ALLOW_EXPERIMENT_FAILURES" == "1" ]]; then
    if ! run_train "$@"; then
      log "failed $1; continuing because ALLOW_EXPERIMENT_FAILURES=1"
    fi
  else
    run_train "$@"
  fi
}

run_train_with_hparams() {
  local label="$1"
  local steps="$2"
  local batch_size="$3"
  local lr="$4"
  local min_lr="$5"
  shift 5

  local BATCH_SIZE="$batch_size"
  local LR="$lr"
  local MIN_LR="$min_lr"
  run_train_allowing_failure "$label" "$steps" "$@"
}

run_lr_sweep() {
  local lr lr_label min_lr
  for lr in $LR_SWEEP_VALUES; do
    lr_label="$(sanitize_label "$lr")"
    min_lr="$MIN_LR"
    case "$lr" in
      1e-4) min_lr="1e-5" ;;
      3e-4) min_lr="3e-5" ;;
      1e-3) min_lr="1e-4" ;;
      3e-3) min_lr="3e-4" ;;
      1e-2) min_lr="1e-3" ;;
      3e-2) min_lr="3e-3" ;;
      1e-1) min_lr="1e-2" ;;
    esac
    run_train_with_hparams "lr_${lr_label}" "$LR_SWEEP_STEPS" "$BATCH_SIZE" "$lr" "$min_lr"
  done
}

run_batch_sweep() {
  local batch batch_label
  for batch in $BATCH_SWEEP_VALUES; do
    batch_label="$(sanitize_label "$batch")"
    run_train_with_hparams "batch_${batch_label}" "$BATCH_SWEEP_STEPS" "$batch" "$LR" "$MIN_LR"
  done
}

run_generate() {
  local label="$1"
  shift
  local extra_args=("$@")
  local run_name checkpoint sample_log
  run_name="${RUN_PREFIX}_${label}"
  checkpoint="runs/checkpoints/${run_name}.pt"
  sample_log="runs/logs/${run_name}_sample.txt"

  if [[ ! -f "$checkpoint" ]]; then
    log "skip generation for $run_name: missing checkpoint $checkpoint"
    return 0
  fi

  local cmd=(
    "$PYTHON_RUN" scripts/generate.py
    --checkpoint "$checkpoint"
    --vocab "$VOCAB_PATH"
    --merges "$MERGES_PATH"
    --prompt "$GENERATE_PROMPT"
    --vocab-size "$VOCAB_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --d-model "$D_MODEL"
    --num-layers "$NUM_LAYERS"
    --num-heads "$NUM_HEADS"
    --d-ff "$D_FF"
    --rope-theta "$ROPE_THETA"
    --max-new-tokens "$GENERATE_MAX_NEW_TOKENS"
    --temperature "$GENERATE_TEMPERATURE"
    --top-p "$GENERATE_TOP_P"
    --device "$DEVICE"
  )
  cmd+=("${extra_args[@]}")

  log "generate sample for $run_name"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" > "$sample_log"
  log "sample written: $sample_log"
}

main() {
  check_runtime

  if [[ "$RUN_BASELINE" == "1" ]]; then
    run_train baseline "$STEPS"
  fi

  if [[ "$RUN_ABLATIONS" == "1" ]]; then
    if require_flag_support no_rmsnorm --no-rmsnorm; then
      run_train no_rmsnorm "$ABLATION_STEPS" --no-rmsnorm
    fi

    if require_flag_support post_norm --norm-position; then
      run_train post_norm "$ABLATION_STEPS" --norm-position post
    fi

    if require_flag_support no_rope --no-rope; then
      run_train no_rope "$ABLATION_STEPS" --no-rope
    fi

    if require_flag_support silu_ffn --ffn-type --silu-d-ff; then
      run_train silu_ffn "$ABLATION_STEPS" --ffn-type silu --silu-d-ff "${SILU_D_FF:-2016}"
    fi
  fi

  if [[ "$RUN_LR_SWEEP" == "1" ]]; then
    run_lr_sweep
  fi

  if [[ "$RUN_BATCH_SWEEP" == "1" ]]; then
    run_batch_sweep
  fi

  if [[ "$RUN_GENERATION" == "1" ]]; then
    run_generate baseline
  fi
}

main "$@"
