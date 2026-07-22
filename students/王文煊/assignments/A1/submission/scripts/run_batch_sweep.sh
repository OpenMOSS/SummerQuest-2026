#!/usr/bin/env bash
# Batch-size sweep on TinyStories (short runs) to study throughput vs. batch size.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}
GPU=${1:-0}
STEPS=600
mkdir -p logs_work/batch_size

for BS in 1 16 64 128 256 512; do
  echo "=== batch_size=${BS} on GPU${GPU} ==="
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} $PY scripts/train.py \
    --train-data artifacts/tinystories_train.npy --val-data artifacts/tinystories_valid.npy \
    --vocab-size 10000 --context-length 256 --d-model 512 --d-ff 1344 --num-layers 4 --num-heads 16 \
    --batch-size ${BS} --total-steps ${STEPS} --warmup-steps 50 \
    --lr 3e-3 --min-lr 3e-4 --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --grad-clip 1.0 \
    --device cuda --amp 1 --eval-every 300 --eval-batches 20 --log-every 20 \
    --log-file logs_work/batch_size/bs_${BS}.jsonl \
    --summary-file logs_work/batch_size/bs_${BS}_summary.json \
    > logs_work/batch_size/bs_${BS}.stdout 2>&1
done
echo "=== BATCH SWEEP DONE ==="
