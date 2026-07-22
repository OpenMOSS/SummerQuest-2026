#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
DOWNLOAD_OWT="${DOWNLOAD_OWT:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

download_if_missing() {
  local url="$1"
  local output="$2"
  if [[ -f "$output" ]]; then
    echo "exists: $DATA_DIR/$output"
    return
  fi
  echo "downloading: $url"
  wget -O "$output" "$url"
}

download_if_missing \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt" \
  "TinyStoriesV2-GPT4-train.txt"

download_if_missing \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" \
  "TinyStoriesV2-GPT4-valid.txt"

if [[ "$DOWNLOAD_OWT" == "1" ]]; then
  download_if_missing \
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz" \
    "owt_train.txt.gz"
  if [[ ! -f "owt_train.txt" ]]; then
    gunzip -k "owt_train.txt.gz"
  fi

  download_if_missing \
    "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz" \
    "owt_valid.txt.gz"
  if [[ ! -f "owt_valid.txt" ]]; then
    gunzip -k "owt_valid.txt.gz"
  fi
fi

echo "done: $DATA_DIR"
