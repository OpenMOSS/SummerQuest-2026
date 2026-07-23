#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_SH="${CONDA_SH:-/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-fourier-llama}"
GPU_ID="${GPU_ID:-0}"
TRAIN_STEPS_CONFIG="${TRAIN_STEPS_CONFIG:-configs/owt_2g_baseline.json}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; stop before encoding/training.")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {props.name}, VRAM GiB: {props.total_memory / 1024**3:.2f}")
PY

TOKENIZER="data/tokenizers/owt_2g_32k"
TRAIN_TEXT="data/owt_train_2g_fallback.txt"
VALID_TEXT="data/owt_valid.txt"
TRAIN_BIN="data/encoded/owt_2g_train.bin"
VALID_BIN="data/encoded/owt_2g_valid.bin"
RUN_DIR="runs/owt_2g_baseline"
CONFIG="$TRAIN_STEPS_CONFIG"

for path in "$TOKENIZER/vocab.json" "$TOKENIZER/merges.json" "$TRAIN_TEXT" "$VALID_TEXT"; do
    if [[ ! -f "$path" ]]; then
        echo "Required input is missing: $path" >&2
        exit 1
    fi
done

mkdir -p data/encoded runs

if [[ ! -f "$TRAIN_BIN" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/encode_dataset.py \
        --input "$TRAIN_TEXT" \
        --tokenizer "$TOKENIZER" \
        --output "$TRAIN_BIN" \
        --dtype uint16 \
        --workers "${ENCODE_WORKERS:-8}"
fi

if [[ ! -f "$VALID_BIN" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/encode_dataset.py \
        --input "$VALID_TEXT" \
        --tokenizer "$TOKENIZER" \
        --output "$VALID_BIN" \
        --dtype uint16 \
        --workers "${ENCODE_WORKERS:-8}"
fi

if [[ ! -f "$CONFIG" ]]; then
    cp configs/tinystories_baseline.json "$CONFIG"
    sed -i 's/"vocab_size": 10000/"vocab_size": 32000/' "$CONFIG"
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/train_lm.py \
    --config "$CONFIG" \
    --train-data "$TRAIN_BIN" \
    --valid-data "$VALID_BIN" \
    --output-dir "$RUN_DIR" \
    --data-dtype uint16 \
    --device cuda

CUDA_VISIBLE_DEVICES="$GPU_ID" python scripts/generate.py \
    --config "$CONFIG" \
    --checkpoint "$RUN_DIR/checkpoint_latest.pt" \
    --tokenizer "$TOKENIZER" \
    --prompt "The government announced" \
    --max-new-tokens 256 \
    --temperature 0.8 \
    --top-p 0.9 \
    --seed 42 \
    --device cuda \
    --output "$RUN_DIR/generation.txt"

echo "OWT 2 GiB fallback training and generation completed."
echo "Summary: $RUN_DIR/summary.json"
echo "Generation: $RUN_DIR/generation.txt"
