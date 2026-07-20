#!/usr/bin/env bash
set -euo pipefail
if command -v python >/dev/null 2>&1; then PYTHON=(python); else PYTHON=(uv run python); fi
exec "${PYTHON[@]}" scripts/evaluate_checkpoint.py --config configs/train_tinystories.json --output logs/train_tinystories_full_val.json "$@"
