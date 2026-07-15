#!/usr/bin/env bash
# Learning-rate sweep on TinyStories (short 2000-step runs), including a divergent run.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${1:-0}
STEPS=2000
mkdir -p logs_work/lr_sweep

for LR in 1e-4 3e-4 1e-3 3e-3 6e-3 1e-2; do
  TAG=${LR}
  echo "=== LR sweep lr=${LR} on GPU${GPU} ==="
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} $PY scripts/train.py \
    --train-data artifacts/tinystories_train.npy --val-data artifacts/tinystories_valid.npy \
    --vocab-size 10000 --context-length 256 --d-model 512 --d-ff 1344 --num-layers 4 --num-heads 16 \
    --batch-size 128 --total-steps ${STEPS} --warmup-steps 100 \
    --lr ${LR} --min-lr $(python3 -c "print(${LR}*0.1)") \
    --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --grad-clip 1.0 \
    --device cuda --amp 1 --eval-every 400 --eval-batches 40 --log-every 20 \
    --log-file logs_work/lr_sweep/lr_${TAG}.jsonl \
    --summary-file logs_work/lr_sweep/lr_${TAG}_summary.json \
    > logs_work/lr_sweep/lr_${TAG}.stdout 2>&1
done
echo "=== LR SWEEP DONE ==="
