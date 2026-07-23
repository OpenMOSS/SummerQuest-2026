#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

PYTHON_RUN="${PYTHON_RUN:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
RUN_PREFIX="${RUN_PREFIX:-tinystories_gpu_target45}"
FORCE="${FORCE:-0}"
RESUME="${RESUME:-1}"

TRAIN_DATA="${TRAIN_DATA:-runs/encoded/tinystories_main_cpu_full_tokenizer_train.bin}"
VALID_DATA="${VALID_DATA:-runs/encoded/tinystories_main_cpu_full_tokenizer_valid.bin}"
BASELINE_CKPT="${BASELINE_CKPT:-runs/checkpoints/tinystories_gpu_baseline.pt}"
HIGH_LR_CKPT="${HIGH_LR_CKPT:-runs/checkpoints/tinystories_gpu_lr_3em3.pt}"

VOCAB_SIZE="${VOCAB_SIZE:-10000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-16}"
D_FF="${D_FF:-1344}"
ROPE_THETA="${ROPE_THETA:-10000.0}"

BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-50}"
EVAL_EVERY="${EVAL_EVERY:-500}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
SEED="${SEED:-0}"
NUM_THREADS="${NUM_THREADS:-}"
TF32="${TF32:-1}"
COMPILE="${COMPILE:-0}"

# Candidate groups:
# - continue20k: conservative continuation from the 10k baseline.
# - continue30k: longer low-LR continuation if 20k is not enough.
# - highlr10k: resume the best short LR-sweep run and continue it to 10k.
# - highlr15k: continue high-LR candidate further with a lower final LR.
CANDIDATES="${CANDIDATES:-continue20k highlr10k}"

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

check_runtime() {
  require_file "$PYTHON_RUN"
  require_file "$TRAIN_DATA"
  require_file "$VALID_DATA"
  require_file "$BASELINE_CKPT"

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

best_resume_path() {
  local output="$1"
  local seed_resume="$2"
  local resume_path

  resume_path=""
  if [[ "$RESUME" == "1" ]]; then
    if [[ -f "$output" ]]; then
      resume_path="$output"
    else
      resume_path="$(latest_step_checkpoint "$output")"
      if [[ -z "$resume_path" ]]; then
        resume_path="$seed_resume"
      fi
    fi
  fi
  printf '%s\n' "$resume_path"
}

run_candidate() {
  local label="$1"
  local seed_resume="$2"
  local steps="$3"
  local lr="$4"
  local min_lr="$5"
  local warmup_steps="$6"

  require_file "$seed_resume"

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

  resume_path="$(best_resume_path "$output" "$seed_resume")"
  if [[ -z "$resume_path" ]]; then
    die "no resume path available for $run_name"
  fi

  cat > "$manifest" <<EOF
RUN_NAME='$run_name'
LABEL='$label'
TRAIN_DATA='$TRAIN_DATA'
VALID_DATA='$VALID_DATA'
SEED_RESUME='$seed_resume'
RESUME_PATH='$resume_path'
VOCAB_SIZE='$VOCAB_SIZE'
CONTEXT_LENGTH='$CONTEXT_LENGTH'
D_MODEL='$D_MODEL'
NUM_LAYERS='$NUM_LAYERS'
NUM_HEADS='$NUM_HEADS'
D_FF='$D_FF'
ROPE_THETA='$ROPE_THETA'
BATCH_SIZE='$BATCH_SIZE'
STEPS='$steps'
LR='$lr'
MIN_LR='$min_lr'
WARMUP_STEPS='$warmup_steps'
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
    --lr "$lr"
    --min-lr "$min_lr"
    --warmup-steps "$warmup_steps"
    --grad-clip "$GRAD_CLIP"
    --log-every "$LOG_EVERY"
    --eval-every "$EVAL_EVERY"
    --eval-batches "$EVAL_BATCHES"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --log-file "$log_file"
    --seed "$SEED"
    --device "$DEVICE"
    --resume "$resume_path"
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

  log "start $run_name from $resume_path"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}" 2>&1 | tee "$console_log"
  log "done $run_name"
}

run_named_candidate() {
  local candidate="$1"
  case "$candidate" in
    continue20k)
      run_candidate continue20k "$BASELINE_CKPT" 20000 1e-4 1e-5 100
      ;;
    continue30k)
      run_candidate continue30k "$BASELINE_CKPT" 30000 6e-5 1e-5 100
      ;;
    highlr10k)
      require_file "$HIGH_LR_CKPT"
      run_candidate highlr10k "$HIGH_LR_CKPT" 10000 3e-3 3e-4 100
      ;;
    highlr15k)
      require_file "$HIGH_LR_CKPT"
      run_candidate highlr15k "$HIGH_LR_CKPT" 15000 1e-3 1e-4 100
      ;;
    *)
      die "unknown candidate: $candidate"
      ;;
  esac
}

main() {
  check_runtime
  local candidate
  for candidate in $CANDIDATES; do
    run_named_candidate "$candidate"
  done
}

main "$@"
