#!/bin/bash

export PYTHONPATH=/remote-home1/wlma/work/assignment1-basics:$PYTHONPATH

# 学习率列表（可根据需要增减）
LR_LIST=("1e-4" "1e-3"  "1e-2" "1e-1" "1e0" "10") 

for LR in "${LR_LIST[@]}"; do
    echo "===== Training with lr=${LR} ====="
    OUT_DIR="data/checkpoints/lr_sweep/lr_${LR}"
    LOG_FILE="log/lr_sweep/lr_${LR}.jsonl"
    mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"

    python scripts/train_tinystories.py \
        --train_data_path data/tinystories_tokenizer/train_tokens.npy \
        --val_data_path data/tinystories_tokenizer/val_tokens.npy \
        --out_dir "$OUT_DIR" \
        --log_dir log/lr_sweep \
        --log_name "tr_${LR}_log.jsonl" \
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --d_ff 1344 \
        --theta 10000.0 \
        --batch_size 128 \
        --total_steps 3000 \
        --lr "$LR" \
        --max_lr "$LR" \
        --min_lr 1e-4 \
        --weight_decay 0.01 \
        --warmup_iters 100 \
        --grad_clip 1.0 \
        --log_interval 100 \
        --val_interval 500 \
        --save_interval 1000 \
        --val_num_batches 20 \
        --device cuda:0

    echo "===== Finished lr=${LR} ====="
done