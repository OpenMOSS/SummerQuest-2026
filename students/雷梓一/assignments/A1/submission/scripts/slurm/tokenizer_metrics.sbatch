#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=a1_tok_metrics
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
python -m uv run python scripts/tokenizer_metrics.py "$@"
