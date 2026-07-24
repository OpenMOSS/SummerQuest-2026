#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --job-name=a1_batch_limit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -m uv run python scripts/benchmark_batch_sizes.py "$@"
