#!/bin/bash
export PYTHONPATH=/remote-home1/wlma/work/assignment1-basics:$PYTHONPATH

BS_LIST=("32" "64" "128" "256")

for BS in "${BS_LIST[@]}"; do
    echo "===== Training with batch_size=${BS} ====="
    OUT_DIR="data/checkpoints/batch_sweep/bs_${BS}"
    mkdir -p "$OUT_DIR" "log/batch_sweep"

    python scripts/train_tinystories.py \
        --train_data_path data/tinystories_tokenizer/train_tokens.npy \
        --val_data_path data/tinystories_tokenizer/val_tokens.npy \
        --out_dir "$OUT_DIR" \
        --log_dir log/batch_sweep \
        --log_name "bs_${BS}_log.jsonl" \
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --d_ff 1344 \
        --theta 10000.0 \
        --batch_size "$BS" \
        --total_steps 3000 \
        --lr 1e-3 \
        --max_lr 1e-3 \
        --min_lr 1e-4 \
        --weight_decay 0.01 \
        --warmup_iters 100 \
        --grad_clip 1.0 \
        --log_interval 100 \
        --val_interval 500 \
        --save_interval 1000 \
        --val_num_batches 20 \
        --device cuda:0

    echo "===== Finished batch_size=${BS} ====="
done