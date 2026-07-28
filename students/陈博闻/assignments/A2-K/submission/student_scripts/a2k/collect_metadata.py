from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import torch

from student_scripts.a2k.common import ALLOCATOR_LIMIT_MIB, public_gpu_metadata, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/run_metadata.json"))
    args = parser.parse_args()
    try:
        triton_version = subprocess.check_output(["python", "-c", "import triton; print(triton.__version__)"], text=True).strip()
    except Exception:
        triton_version = "unavailable"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    payload = public_gpu_metadata()
    payload.update(
        {
            "assignment": "A2-K",
            "assignment_version": "26.1.4-k-rc.3",
            "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
            "working_commit": commit,
            "seed": {
                "checkpointing_model": 0,
                "checkpointing_data": 1,
                "attention_inputs": 0,
                "small_model": 10,
                "small_model_data": 11,
                "correctness": [0, 1, 2],
            },
            "commands": [
                "python -m pytest tests/test_attention.py -v",
                "python -m student_scripts.a2k.correctness --include-triton --output local_results/a2k/correctness.json",
                "python -m student_scripts.a2k.benchmark_checkpointing --output local_results/a2k/checkpointing.csv",
                "python -m student_scripts.a2k.benchmark_attention --output-dir local_results/a2k",
                "python -m student_scripts.a2k.collect_metadata --output local_results/a2k/run_metadata.json",
                "python -m student_scripts.a2k.memory_evidence --results-dir local_results/a2k --output local_results/a2k/memory_evidence.json",
                "python -m student_scripts.a2k.plot_results --results-dir local_results/a2k --assets-dir local_results/a2k/assets",
            ],
            "triton": triton_version,
            "measurement_timer": "triton.testing.do_bench or CUDA events with synchronization",
            "attention_warmup_ms": 100,
            "attention_rep_ms": 300,
            "attention_quantiles": [0.2, 0.5, 0.8],
            "checkpoint_warmup_steps": 3,
            "checkpoint_measurement_steps": 5,
            "hard_limit_mib": 24 * 1024,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
            "notes": "Do not include private machine identifiers, internal paths, process lists, or credentials.",
        }
    )
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        payload["allocator_fraction"] = min(1.0, (ALLOCATOR_LIMIT_MIB * 1024**2) / total)
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
