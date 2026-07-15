#!/usr/bin/env bash
set -euo pipefail
if command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python)
elif command -v python >/dev/null 2>&1; then
  PYTHON=(python)
else
  PYTHON=(python3)
fi
exec "${PYTHON[@]}" scripts/tokenization_experiments.py suite --config configs/tokenizer_experiments.json "$@"
