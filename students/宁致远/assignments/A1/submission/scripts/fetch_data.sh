#!/usr/bin/env bash
# Download TinyStories and OpenWebText (already-tokenized-style txt) to $DATA_ROOT.
# Idempotent: skips files already present.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/inspire/hdd/global_user/heziwei-25044/zy-ning/models/a1_data}
mkdir -p "$DATA_ROOT/tinystories" "$DATA_ROOT/owt"

# TinyStories: use the CS336-hosted mirror (single-file txt).
TS_URL_TRAIN=${TS_URL_TRAIN:-https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt}
TS_URL_VAL=${TS_URL_VAL:-https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt}

fetch() {
  local url="$1"; local out="$2"
  if [ -s "$out" ]; then
    echo "have: $out ($(du -h "$out" | cut -f1))"
  else
    echo "fetching: $out"
    curl -L --fail --retry 3 -o "$out.part" "$url"
    mv "$out.part" "$out"
  fi
}

# HF mirror if HF_ENDPOINT is set; otherwise honor HTTPS_PROXY as-is.
if [ -n "${HF_ENDPOINT:-}" ]; then
  TS_URL_TRAIN=${TS_URL_TRAIN/https:\/\/huggingface.co/$HF_ENDPOINT}
  TS_URL_VAL=${TS_URL_VAL/https:\/\/huggingface.co/$HF_ENDPOINT}
fi

fetch "$TS_URL_TRAIN" "$DATA_ROOT/tinystories/train.txt"
fetch "$TS_URL_VAL"   "$DATA_ROOT/tinystories/val.txt"

# OpenWebText: use the CS336 provided split (owt_train.txt / owt_valid.txt).
# The stanford-cs336 course hosts these; if unreachable, users can override URLs.
OWT_URL_TRAIN=${OWT_URL_TRAIN:-https://huggingface.co/datasets/stanford-cs336/owt-preprocessed/resolve/main/owt_train.txt}
OWT_URL_VAL=${OWT_URL_VAL:-https://huggingface.co/datasets/stanford-cs336/owt-preprocessed/resolve/main/owt_valid.txt}
if [ -n "${HF_ENDPOINT:-}" ]; then
  OWT_URL_TRAIN=${OWT_URL_TRAIN/https:\/\/huggingface.co/$HF_ENDPOINT}
  OWT_URL_VAL=${OWT_URL_VAL/https:\/\/huggingface.co/$HF_ENDPOINT}
fi
fetch "$OWT_URL_TRAIN" "$DATA_ROOT/owt/train.txt" || echo "(OWT train not fetched; override OWT_URL_TRAIN)"
fetch "$OWT_URL_VAL"   "$DATA_ROOT/owt/val.txt"   || echo "(OWT val not fetched; override OWT_URL_VAL)"

echo "done. root: $DATA_ROOT"
