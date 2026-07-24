#!/usr/bin/env bash
set -euo pipefail

: "${A1_PYTHON:?Set A1_PYTHON to a CUDA-compatible Python executable}"

cd "$(dirname "$0")/.."

launch() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  local output_dir="runs/$name"

  if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite existing run: $output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=. \
    "$A1_PYTHON" scripts/train_lm.py \
      --config "$config" \
      --train-data data/encoded/tinystories_train.bin \
      --valid-data data/encoded/tinystories_valid.bin \
      --output-dir "$output_dir" \
      --data-dtype uint16 \
      --device cuda \
      >"$output_dir/console.log" 2>&1 &
  echo "gpu=$gpu pid=$! run=$name config=$config"
}

launch 1 tinystories_ablation_no_rmsnorm configs/tinystories_ablation_no_rmsnorm.json
launch 2 tinystories_ablation_postnorm configs/tinystories_ablation_postnorm.json
launch 3 tinystories_ablation_nope configs/tinystories_ablation_nope.json
launch 4 tinystories_ablation_silu configs/tinystories_ablation_silu.json
launch 5 tinystories_lr_1e-4 configs/tinystories_lr_1e-4.json
launch 6 tinystories_lr_1e-3 configs/tinystories_lr_1e-3.json
launch 7 tinystories_lr_1e-2 configs/tinystories_lr_1e-2.json
