#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=a1_generate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
python -m uv run python scripts/generate.py "$@"
