#!/usr/bin/env bash
set -euo pipefail
if command -v python >/dev/null 2>&1; then PYTHON=(python); else PYTHON=(uv run python); fi
exec "${PYTHON[@]}" scripts/train.py --config configs/train_tinystories.json "$@"
