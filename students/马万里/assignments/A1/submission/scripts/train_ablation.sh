#!/bin/bash
export PYTHONPATH=/remote-home1/wlma/work/assignment1-basics:$PYTHONPATH

# 消融实验配置
EXPERIMENTS=(
    "no_rmsnorm"
    "post_norm"
    "nope"
    "silu_ffn"
)

TOTAL_STEPS=5000  # 可根据需要调整

for EXP in "${EXPERIMENTS[@]}"; do
    echo "===== Running ablation: ${EXP} ====="

    OUT_DIR="data/checkpoints/ablation_${EXP}"
    LOG_DIR="log/ablation"
    mkdir -p "$OUT_DIR" "$LOG_DIR"

    if [ "$EXP" == "silu_ffn" ]; then
        # SiLU 替代 SwiGLU 使用 --use_silu_ffn，其他参数默认（pre_norm）
        EXTRA_ARGS="--use_silu_ffn"
        BLOCK_TYPE="pre_norm"
    else
        # 其他三个消融分别用不同的 block_type
        EXTRA_ARGS=""
        BLOCK_TYPE="$EXP"
    fi

    python scripts/train_tinystories.py \
        --train_data_path data/tinystories_tokenizer/train_tokens.npy \
        --val_data_path data/tinystories_tokenizer/val_tokens.npy \
        --out_dir "$OUT_DIR" \
        --log_dir "$LOG_DIR" \
        --log_name "ablation_${EXP}_log.jsonl"\
        --vocab_size 10000 \
        --context_length 256 \
        --d_model 512 \
        --num_layers 4 \
        --num_heads 16 \
        --d_ff 1344 \
        --theta 10000.0 \
        --batch_size 128 \
        --total_steps "$TOTAL_STEPS" \
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
        --device cuda:0 \
        --block_type "$BLOCK_TYPE" \
        $EXTRA_ARGS

    echo "===== Finished ablation: ${EXP} ====="
done