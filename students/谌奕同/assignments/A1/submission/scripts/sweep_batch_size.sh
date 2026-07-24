#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_BASE="outputs/batch_sweep"
mkdir -p "$OUTPUT_BASE"

BATCH_SIZES=(1 64 128 256 512)
CTX=256

# Small batches are very slow; use fewer steps for bs=1.
declare -A STEPS_MAP
STEPS_MAP=(
    [1]=200
    [64]=1000
    [128]=1000
    [256]=1000
    [512]=1000
)

cat > "$OUTPUT_BASE/summary.txt" <<EOF
Batch Size Sweep
Context length: $CTX
Steps vary by batch size (bs=1 uses 200, others use 1000)
LR: 1e-3
EOF

for BS in "${BATCH_SIZES[@]}"; do
    OUT="$OUTPUT_BASE/bs_${BS}"
    mkdir -p "$OUT"
    STEPS=${STEPS_MAP[$BS]}
    echo "=== Running batch_size=$BS (steps=$STEPS) ==="
    set +e
    uv run python cs336_basics/train.py \
        --train_tokens outputs/tinystories/train.npy \
        --val_tokens outputs/tinystories/val.npy \
        --vocab_path outputs/tinystories/vocab.json \
        --merges_path outputs/tinystories/merges.txt \
        --output_dir "$OUT" \
        --vocab_size 10000 --context_length $CTX --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
        --batch_size "$BS" --max_iters "$STEPS" --learning_rate 0.001 --min_learning_rate 6e-05 --warmup_iters 100 \
        --eval_interval 200 --eval_batches 10 --checkpoint_interval 100000 --log_interval 100 \
        --device cuda --seed 42 \
        > "$OUT/run.log" 2>&1
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ]; then
        BEST=$(grep "Best val loss" "$OUT/train.log" | awk '{print $NF}')
        TIME=$(grep -oP 'time [0-9.]+s' "$OUT/train.log" | tail -n 1 | sed 's/time //')
        echo "bs=$BS -> best val loss $BEST, time $TIME" | tee -a "$OUTPUT_BASE/summary.txt"
    else
        echo "bs=$BS -> OOM/FAILED (exit $EXIT_CODE)" | tee -a "$OUTPUT_BASE/summary.txt"
    fi
done

echo "Sweep complete. Summary: $OUTPUT_BASE/summary.txt"
