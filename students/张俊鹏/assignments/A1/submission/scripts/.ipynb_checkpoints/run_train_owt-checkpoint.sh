#!/bin/bash
set -e

# === Config ===
PYTHON=/opt/conda/bin/python
ROOT=/inspire/ssd/project/cq-scientific-cooperation-zone/tongjingqi-CZXS25110029/zjp_code
cd "$ROOT/summerQuest/assignment1-basics"

# Data
TRAIN_DATA=${TRAIN_DATA:-"$ROOT/summerQuest/assignment1-basics/data/tinystories/tinystories_train.npy"}
VAL_DATA=${VAL_DATA:-"$ROOT/summerQuest/assignment1-basics/data/tinystories/tinystories_val.npy"}

# Model (per 7.2 spec)
VOCAB_SIZE=${VOCAB_SIZE:-10000}
CONTEXT_LENGTH=256
D_MODEL=512
D_FF=1344
NUM_LAYERS=4
NUM_HEADS=16
THETA=10000.0

# Training (batch_size * total_steps * context_length = 327,680,000)
BATCH_SIZE=${BATCH_SIZE:-64}
TOTAL_STEPS=${TOTAL_STEPS:-20000}
EVAL_EVERY=${EVAL_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-10}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-2000}

# Optimizer defaults
MAX_LR=${MAX_LR:-1e-3}
MIN_LR=${MIN_LR:-1e-4}
WARMUP_ITERS=${WARMUP_ITERS:-100}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
BETA1=${BETA1:-0.9}
BETA2=${BETA2:-0.95}
EPS=${EPS:-1e-8}

# Experiment
EXP_NAME=${EXP_NAME:-"baseline"}
EXP_DIR=${EXP_DIR:-"experiments"}
CKPT_DIR=${CKPT_DIR:-"checkpoints/$EXP_NAME"}

echo "============================================"
echo "Experiment: $EXP_NAME"
echo "============================================"
echo "Config:"
echo "  batch_size=$BATCH_SIZE  total_steps=$TOTAL_STEPS  context=$CONTEXT_LENGTH"
echo "  tokens = $((BATCH_SIZE * TOTAL_STEPS * CONTEXT_LENGTH))"
echo "  max_lr=$MAX_LR  min_lr=$MIN_LR  warmup=$WARMUP_ITERS"
echo "  weight_decay=$WEIGHT_DECAY  beta1=$BETA1  beta2=$BETA2"
echo "============================================"

"$PYTHON" -u -m scripts.train \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --vocab_size $VOCAB_SIZE \
    --context_length $CONTEXT_LENGTH \
    --d_model $D_MODEL \
    --d_ff $D_FF \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --theta $THETA \
    --batch_size $BATCH_SIZE \
    --total_steps $TOTAL_STEPS \
    --eval_every $EVAL_EVERY \
    --log_every $LOG_EVERY \
    --checkpoint_every $CHECKPOINT_EVERY \
    --max_lr $MAX_LR \
    --min_lr $MIN_LR \
    --warmup_iters $WARMUP_ITERS \
    --weight_decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --beta2 $BETA2 \
    --eps $EPS \
    --exp_name "$EXP_NAME" \
    --exp_dir "$EXP_DIR" \
    --checkpoint_dir "$CKPT_DIR" \
    --device cuda \
    --compile \
    --wandb

echo "Done: $EXP_NAME"