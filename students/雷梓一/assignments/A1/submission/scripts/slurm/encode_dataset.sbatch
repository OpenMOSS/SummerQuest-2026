#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=a1_encode
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
python -m uv run python scripts/encode_dataset.py "$@"
