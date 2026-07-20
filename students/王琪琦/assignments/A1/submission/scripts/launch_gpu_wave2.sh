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

launch 0 tinystories_batch_1 configs/tinystories_batch_1.json
launch 1 tinystories_batch_64 configs/tinystories_batch_64.json
launch 2 tinystories_batch_256 configs/tinystories_batch_256.json
launch 3 tinystories_batch_512 configs/tinystories_batch_512.json

probe_output="runs/tinystories_batch_probe"
if [[ -e "$probe_output" ]]; then
  echo "refusing to overwrite existing run: $probe_output" >&2
  exit 1
fi
mkdir -p "$probe_output"
nohup env \
  CUDA_VISIBLE_DEVICES=4 \
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. \
  "$A1_PYTHON" scripts/probe_batch_size.py \
    --config configs/tinystories_baseline.json \
    --output "$probe_output/report.json" \
    --batches 128 256 512 1024 \
    --device cuda \
    >"$probe_output/console.log" 2>&1 &
echo "gpu=4 pid=$! run=tinystories_batch_probe"

launch 5 tinystories_lr_1e-1 configs/tinystories_lr_1e-1.json

generation_dir="runs/generations"
if [[ -e "$generation_dir" ]]; then
  echo "refusing to overwrite existing run: $generation_dir" >&2
  exit 1
fi
mkdir -p "$generation_dir"
nohup env CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. \
  "$A1_PYTHON" scripts/generate.py \
    --config configs/tinystories_baseline.json \
    --checkpoint runs/tinystories_baseline/checkpoint_latest.pt \
    --tokenizer data/tokenizers/tinystories_10k \
    --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cuda \
    >"$generation_dir/baseline.txt" 2>&1 &
echo "gpu=6 pid=$! run=generation_baseline"

nohup env CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. \
  "$A1_PYTHON" scripts/generate.py \
    --config configs/tinystories_lr_1e-2.json \
    --checkpoint runs/tinystories_lr_1e-2/checkpoint_latest.pt \
    --tokenizer data/tokenizers/tinystories_10k \
    --prompt "Once upon a time" \
    --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cuda \
    >"$generation_dir/lr_1e-2.txt" 2>&1 &
echo "gpu=7 pid=$! run=generation_lr_1e-2"
