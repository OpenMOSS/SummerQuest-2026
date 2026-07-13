#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

scripts/download_a1_data.sh
scripts/prepare_tinystories.sh
scripts/train_tinystories_baseline.sh
scripts/generate_tinystories_sample.sh | tee "${GEN_OUT:-runs/tinystories_baseline/sample.txt}"
