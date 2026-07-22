#!/usr/bin/env bash
# Architecture ablations on TinyStories. All runs share the same 5000-step budget
# and schedule so their learning curves are directly comparable.
# Usage: run_ablations.sh <gpu> <ablation1> [ablation2 ...]
# ablations: baseline no_rmsnorm no_rmsnorm_lowlr post_norm nope silu
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${1:-0}; shift
STEPS=5000
mkdir -p logs_work/ablation

common=(--train-data artifacts/tinystories_train.npy --val-data artifacts/tinystories_valid.npy
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16
  --batch-size 128 --total-steps ${STEPS} --warmup-steps 100 --min-lr 3e-4
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --grad-clip 1.0
  --device cuda --amp 1 --eval-every 250 --eval-batches 40 --log-every 20)

run() {
  local name=$1; shift
  echo "=== ablation ${name} on GPU${GPU} ==="
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} $PY scripts/train.py "${common[@]}" "$@" \
    --log-file logs_work/ablation/ablation_${name}.jsonl \
    --summary-file logs_work/ablation/ablation_${name}_summary.json \
    > logs_work/ablation/ablation_${name}.stdout 2>&1
}

for ab in "$@"; do
  case $ab in
    baseline)         run baseline --d-ff 1344 --lr 3e-3 --norm rmsnorm --norm-position pre --use-rope 1 --ffn swiglu ;;
    no_rmsnorm)       run no_rmsnorm --d-ff 1344 --lr 3e-3 --norm none --norm-position pre --use-rope 1 --ffn swiglu ;;
    no_rmsnorm_lowlr) run no_rmsnorm_lowlr --d-ff 1344 --lr 1e-3 --norm none --norm-position pre --use-rope 1 --ffn swiglu ;;
    post_norm)        run post_norm --d-ff 1344 --lr 3e-3 --norm rmsnorm --norm-position post --use-rope 1 --ffn swiglu ;;
    nope)             run nope --d-ff 1344 --lr 3e-3 --norm rmsnorm --norm-position pre --use-rope 0 --ffn swiglu ;;
    silu)             run silu --d-ff 2048 --lr 3e-3 --norm rmsnorm --norm-position pre --use-rope 1 --ffn silu ;;
    *) echo "unknown ablation: $ab" ;;
  esac
done
echo "=== ABLATIONS DONE (gpu ${GPU}): $* ==="
