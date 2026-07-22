#!/usr/bin/env bash
# End-to-end GPU-side runner. Expects tokens under $TOKENS (already produced on CPU box):
#   $TOKENS/ts_train.bin, ts_val.bin  (10K vocab)
#   $TOKENS/owt_train.bin, owt_val.bin (32K vocab, optional)
# and tokenizers at $TOK_TS, $TOK_OWT (dirs containing tokenizer.pkl).
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
TOKENS=${TOKENS:-$ROOT/runs/tokens}
TOK_TS=${TOK_TS:-$ROOT/runs/tokenizer_ts}
TOK_OWT=${TOK_OWT:-$ROOT/runs/tokenizer_owt}
OUT=${OUT_ROOT:-$ROOT/runs}
PY=${PY:-$ROOT/.venv/bin/python}
DEVICE=${DEVICE:-cuda}
mkdir -p "$OUT"

TS="--train $TOKENS/ts_train.bin --val $TOKENS/ts_val.bin"
TS_ARGS="--vocab-size 10000 --context-length 256 --d-model 512 --d-ff 1344 \
         --num-layers 4 --num-heads 16 --rope-theta 10000 \
         --lr 3e-4 --min-lr 3e-5 --warmup 200 --weight-decay 0.1 \
         --beta1 0.9 --beta2 0.95 --grad-clip 1.0 \
         --log-interval 50 --val-interval 500 --ckpt-interval 2000 \
         --val-iters 20 --device $DEVICE --dtype bfloat16"

# 1) TinyStories baseline (10k steps, batch 128)
$PY -m cs336_basics.train $TS $TS_ARGS \
    --batch-size 128 --total-steps 10000 --out "$OUT/ts_baseline"

# 2) LR sweep (2k steps each, keep one clearly diverging)
for lr in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  $PY -m cs336_basics.train $TS $TS_ARGS \
      --batch-size 128 --total-steps 2000 --warmup 100 --lr "$lr" \
      --min-lr "$(python3 -c "print($lr*0.1)")" \
      --out "$OUT/ts_lr_$lr" || echo "(lr=$lr diverged, keeping partial log)"
done

# 3) Batch-size sweep (2k steps each)
for bs in 1 8 32 64 128; do
  $PY -m cs336_basics.train $TS $TS_ARGS \
      --batch-size "$bs" --total-steps 2000 --warmup 100 \
      --out "$OUT/ts_bs_$bs" || echo "(bs=$bs failed, likely OOM)"
done

# 4) Four ablations (5k steps each)
for v in no_rmsnorm post_norm nope silu_ffn; do
  $PY -m cs336_basics.train $TS $TS_ARGS \
      --batch-size 128 --total-steps 5000 --variant "$v" \
      --out "$OUT/ts_ablation_$v" || echo "(ablation $v failed)"
done

# 5) TinyStories generation samples
$PY -m cs336_basics.scripts sample \
    --ckpt "$OUT/ts_baseline/ckpt-final.pt" --tokenizer "$TOK_TS" \
    --config "$OUT/ts_baseline/config.json" \
    --prompt "Once upon a time" --max-new-tokens 256 \
    --temperature 0.8 --top-p 0.95 --n 3 --device "$DEVICE" \
  > "$OUT/ts_samples.txt"

# 6) OWT baseline (if bin exists)
if [ -s "$TOKENS/owt_train.bin" ]; then
  $PY -m cs336_basics.train \
      --train "$TOKENS/owt_train.bin" --val "$TOKENS/owt_val.bin" \
      --vocab-size 32000 --context-length 256 --d-model 512 --d-ff 1344 \
      --num-layers 4 --num-heads 16 --rope-theta 10000 \
      --batch-size 128 --total-steps 10000 \
      --lr 3e-4 --min-lr 3e-5 --warmup 200 --weight-decay 0.1 \
      --beta1 0.9 --beta2 0.95 --grad-clip 1.0 \
      --log-interval 50 --val-interval 500 --ckpt-interval 2000 \
      --val-iters 20 --device "$DEVICE" --dtype bfloat16 \
      --out "$OUT/owt_baseline"

  $PY -m cs336_basics.scripts sample \
      --ckpt "$OUT/owt_baseline/ckpt-final.pt" --tokenizer "$TOK_OWT" \
      --config "$OUT/owt_baseline/config.json" \
      --prompt "The " --max-new-tokens 256 --temperature 0.8 --top-p 0.95 \
      --n 3 --device "$DEVICE" > "$OUT/owt_samples.txt"
fi

# 7) Plots
$PY scripts/plot_curves.py \
    --runs "baseline=$OUT/ts_baseline/train.jsonl" \
    --out "$OUT/../assets/ts_baseline.png" --title "TinyStories baseline"
$PY scripts/plot_curves.py \
    --runs $(for lr in 1e-4 3e-4 1e-3 3e-3 1e-2; do echo -n "lr=$lr=$OUT/ts_lr_$lr/train.jsonl "; done) \
    --out "$OUT/../assets/lr_sweep.png" --title "LR sweep"
$PY scripts/plot_curves.py \
    --runs $(for bs in 1 8 32 64 128; do echo -n "bs=$bs=$OUT/ts_bs_$bs/train.jsonl "; done) \
    --out "$OUT/../assets/batch_size.png" --title "Batch-size sweep"
$PY scripts/plot_curves.py \
    --runs "baseline=$OUT/ts_baseline/train.jsonl" \
           $(for v in no_rmsnorm post_norm nope silu_ffn; do echo -n "$v=$OUT/ts_ablation_$v/train.jsonl "; done) \
    --out "$OUT/../assets/ablations.png" --title "Architecture ablations"
if [ -s "$OUT/owt_baseline/train.jsonl" ]; then
  $PY scripts/plot_curves.py --runs "owt=$OUT/owt_baseline/train.jsonl" \
      --out "$OUT/../assets/owt_baseline.png" --title "OpenWebText baseline"
fi

echo "all done. see $OUT/"
