#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p logs runs/tinystories_baseline

exec .venv/bin/python scripts/train_lm.py \
  --config configs/tinystories_baseline.json \
  "$@"
