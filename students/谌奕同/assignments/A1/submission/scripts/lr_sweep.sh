#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_BASE="outputs/lr_sweep"
mkdir -p "$OUTPUT_BASE"

LEARNING_RATES=(3e-4 6e-4 1e-3 3e-3 1e-2)
CTX=256
STEPS=1000
BS=128

cat > "$OUTPUT_BASE/summary.txt" <<EOF
Learning Rate Sweep
Context length: $CTX
Batch size: $BS
Steps: $STEPS per LR
EOF

for LR in "${LEARNING_RATES[@]}"; do
    OUT="$OUTPUT_BASE/lr_${LR}"
    mkdir -p "$OUT"
    echo "=== Running learning_rate=$LR ==="
    set +e
    uv run python cs336_basics/train.py \
        --train_tokens outputs/tinystories/train.npy \
        --val_tokens outputs/tinystories/val.npy \
        --vocab_path outputs/tinystories/vocab.json \
        --merges_path outputs/tinystories/merges.txt \
        --output_dir "$OUT" \
        --vocab_size 10000 --context_length $CTX --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
        --batch_size "$BS" --max_iters "$STEPS" --learning_rate "$LR" --min_learning_rate 6e-05 --warmup_iters 100 \
        --eval_interval 200 --eval_batches 10 --checkpoint_interval 100000 --log_interval 100 \
        --device cuda --seed 42 \
        > "$OUT/run.log" 2>&1
    EXIT_CODE=$?
    set -e

    if [ "$EXIT_CODE" -eq 0 ] && [ -f "$OUT/train.log" ]; then
        BEST=$(grep "Best val loss" "$OUT/train.log" | awk '{print $NF}')
        echo "lr=$LR -> best val loss $BEST" | tee -a "$OUTPUT_BASE/summary.txt"
    else
        echo "lr=$LR -> DIVERGED/FAILED (exit $EXIT_CODE)" | tee -a "$OUTPUT_BASE/summary.txt"
    fi
done

echo "Sweep complete. Summary: $OUTPUT_BASE/summary.txt"
