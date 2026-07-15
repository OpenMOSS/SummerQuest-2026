#!/usr/bin/env bash
# High learning-rate runs to demonstrate divergence (grad clipping disabled so
# instability is visible). Complements run_lr_sweep.sh.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${1:-0}
STEPS=800
mkdir -p logs_work/lr_sweep

for LR in 2e-2 5e-2 1e-1; do
  echo "=== divergence lr=${LR} on GPU${GPU} ==="
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} $PY scripts/train.py \
    --train-data artifacts/tinystories_train.npy --val-data artifacts/tinystories_valid.npy \
    --vocab-size 10000 --context-length 256 --d-model 512 --d-ff 1344 --num-layers 4 --num-heads 16 \
    --batch-size 128 --total-steps ${STEPS} --warmup-steps 50 \
    --lr ${LR} --min-lr $(python3 -c "print(${LR}*0.1)") \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --grad-clip 1e9 \
    --device cuda --amp 1 --eval-every 400 --eval-batches 20 --log-every 10 \
    --log-file logs_work/lr_sweep/lr_${LR}.jsonl \
    --summary-file logs_work/lr_sweep/lr_${LR}_summary.json \
    > logs_work/lr_sweep/lr_${LR}.stdout 2>&1
done
echo "=== DIVERGENCE RUNS DONE ==="
