#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DEVICE="${DEVICE:-auto}"

export DEVICE

uv sync --frozen --no-install-project

bash scripts/tasks/run_a1.sh test

bash scripts/tasks/run_a1.sh tokenizer_tinystories

bash scripts/tasks/run_a1.sh encode_tinystories

bash scripts/tasks/run_a1.sh train_tinystories

bash scripts/tasks/run_a1.sh generate_tinystories

bash scripts/tasks/run_a1.sh lr_sweep

bash scripts/tasks/run_a1.sh batch_sweep

bash scripts/tasks/run_a1.sh ablations

bash scripts/tasks/run_a1.sh tokenizer_owt

bash scripts/tasks/run_a1.sh encode_owt

bash scripts/tasks/run_a1.sh train_owt

PROMPT="${PROMPT:-The following is a story about a robot:}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.95}"

export PROMPT MAX_NEW_TOKENS TEMPERATURE TOP_P

bash scripts/tasks/run_a1.sh generate_tinystories > "logs/generation_tinystories.txt"

bash scripts/tasks/run_a1.sh generate_owt > "logs/generation_owt.txt"
