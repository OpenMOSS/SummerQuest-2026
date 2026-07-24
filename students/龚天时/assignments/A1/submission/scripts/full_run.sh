#!/bin/bash
BASE="--train_data data/ts_train.npy --val_data data/ts_valid.npy \
  --vocab_size 10000 --context_length 256 \
  --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 --rope_theta 10000 \
  --warmup_iters 800 --max_lr 3e-4 --min_lr 3e-5 --weight_decay 0.1 \
  --eval_interval 1000 --ckpt_interval 10000 --device cuda"
BS32="--batch_size 32 --total_steps 40000"

echo "===== 基线 ====="
uv run python scripts/train.py $BASE $BS32 --out_dir runs/ts_baseline

echo "===== 四消融 ====="
uv run python scripts/train.py $BASE $BS32 --use_rmsnorm false --out_dir runs/ts_nonorm
uv run python scripts/train.py $BASE $BS32 --norm_position post --out_dir runs/ts_postnorm
uv run python scripts/train.py $BASE $BS32 --use_rope false --out_dir runs/ts_norope
uv run python scripts/train.py $BASE $BS32 --ffn_type silu --out_dir runs/ts_silu

echo "===== LR sweep(收敛的跑满,发散的少跑)====="
for lr in 1e-4 1e-3 3e-3; do
  uv run python scripts/train.py $BASE $BS32 --max_lr $lr --out_dir runs/lr_$lr
done
# 发散 run:少步数,加个极端值保证发散
uv run python scripts/train.py $BASE --batch_size 32 --total_steps 3000 --max_lr 1e-2 --out_dir runs/lr_1e-2
uv run python scripts/train.py $BASE --batch_size 32 --total_steps 3000 --max_lr 1e-1 --out_dir runs/lr_1e-1

echo "===== batch 扫(保持 token 量)====="
uv run python scripts/train.py $BASE --batch_size 64 --total_steps 20000 --out_dir runs/bs_64
uv run python scripts/train.py $BASE --batch_size 128 --total_steps 10000 --out_dir runs/bs_128

echo "===== OWT ====="
uv run python scripts/train.py \
  --train_data data/owt_train.npy --val_data data/owt_valid.npy \
  --vocab_size 32000 --context_length 256 \
  --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 --rope_theta 10000 \
  --batch_size 32 --total_steps 40000 --warmup_iters 800 \
  --max_lr 3e-4 --min_lr 3e-5 --weight_decay 0.1 \
  --eval_interval 1000 --ckpt_interval 10000 --device cuda \
  --out_dir runs/owt_baseline