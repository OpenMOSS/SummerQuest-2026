#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PIPELINE_DIR="runs/owt_pipeline"
STATUS_FILE="$PIPELINE_DIR/status.tsv"
mkdir -p "$PIPELINE_DIR"

exec 9>"$PIPELINE_DIR/runner.lock"
if ! flock -n 9; then
    printf '%s\tPIPELINE\talready-running\n' "$(date -Is)" >> "$STATUS_FILE"
    exit 0
fi

rm -f "$PIPELINE_DIR/COMPLETE" "$PIPELINE_DIR/FAILED" "$PIPELINE_DIR/RUNNING"
touch "$PIPELINE_DIR/RUNNING"

finish_failed() {
    exit_code=$?
    printf '%s\tPIPELINE\tfailed:%s\n' "$(date -Is)" "$exit_code" >> "$STATUS_FILE"
    rm -f "$PIPELINE_DIR/RUNNING"
    touch "$PIPELINE_DIR/FAILED"
    exit "$exit_code"
}
trap finish_failed ERR

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/../cache/uv}"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Python environment not found: %s\n' "$PYTHON" >&2
    exit 1
fi
TOKENIZER="data/owt_bpe_32k.json"
TRAIN_TEXT="data/owt_train.txt"
VAL_TEXT="data/owt_valid.txt"
TRAIN_BIN="data/owt_train_uint16.bin"
VAL_BIN="data/owt_valid_uint16.bin"

printf '%s\tPIPELINE\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"

if [[ ! -f "$TOKENIZER" ]]; then
    printf '%s\tBPE\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"
    "$PYTHON" -m cs336_basics.bpe_experiment "$TRAIN_TEXT" "$TOKENIZER" --vocab-size 32000 --special-token '<|endoftext|>' > "$PIPELINE_DIR/bpe.log" 2>&1
    printf '%s\tBPE\tsucceeded\n' "$(date -Is)" >> "$STATUS_FILE"
else
    printf '%s\tBPE\tskipped-existing\n' "$(date -Is)" >> "$STATUS_FILE"
fi

if [[ ! -f "$VAL_BIN" ]]; then
    printf '%s\tENCODE_VALID\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"
    "$PYTHON" scripts/encode_dataset.py "$VAL_TEXT" "$TOKENIZER" "$VAL_BIN" --special-token '<|endoftext|>' > "$PIPELINE_DIR/encode_valid.json"
    printf '%s\tENCODE_VALID\tsucceeded\n' "$(date -Is)" >> "$STATUS_FILE"
else
    printf '%s\tENCODE_VALID\tskipped-existing\n' "$(date -Is)" >> "$STATUS_FILE"
fi

if [[ ! -f "$TRAIN_BIN" ]]; then
    printf '%s\tENCODE_TRAIN\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"
    "$PYTHON" scripts/encode_dataset.py "$TRAIN_TEXT" "$TOKENIZER" "$TRAIN_BIN" --special-token '<|endoftext|>' > "$PIPELINE_DIR/encode_train.json"
    printf '%s\tENCODE_TRAIN\tsucceeded\n' "$(date -Is)" >> "$STATUS_FILE"
else
    printf '%s\tENCODE_TRAIN\tskipped-existing\n' "$(date -Is)" >> "$STATUS_FILE"
fi

printf '%s\tTRAIN\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"
"$PYTHON" scripts/train.py --config configs/owt_baseline_full.json
printf '%s\tTRAIN\tsucceeded\n' "$(date -Is)" >> "$STATUS_FILE"

printf '%s\tGENERATE\tstarted\n' "$(date -Is)" >> "$STATUS_FILE"
"$PYTHON" scripts/generate.py --config configs/owt_baseline_full.json --checkpoint runs/owt_baseline_full/checkpoint.pt --tokenizer "$TOKENIZER" --prompt 'The' --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 > "$PIPELINE_DIR/generation.txt"
printf '%s\tGENERATE\tsucceeded\n' "$(date -Is)" >> "$STATUS_FILE"

tar -czf "$PIPELINE_DIR/results.tar.gz" configs/owt_baseline_full.json "$TOKENIZER" "$PIPELINE_DIR/bpe.log" "$PIPELINE_DIR/encode_valid.json" "$PIPELINE_DIR/encode_train.json" "$PIPELINE_DIR/generation.txt" "$STATUS_FILE" runs/owt_baseline_full/metrics.jsonl
printf '%s\tPIPELINE\tcomplete\n' "$(date -Is)" >> "$STATUS_FILE"
rm -f "$PIPELINE_DIR/RUNNING"
touch "$PIPELINE_DIR/COMPLETE"
trap - ERR
