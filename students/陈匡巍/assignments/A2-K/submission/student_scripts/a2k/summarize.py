"""Combine formal-run metadata and enforce the 24 GiB evidence invariant."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import triton

from student_scripts.a2k.common import (
    ALLOCATOR_LIMIT_MIB,
    HARD_LIMIT_MIB,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--compile-csv", type=Path, required=True)
    parser.add_argument("--flash-csv", type=Path, required=True)
    parser.add_argument("--correctness-json", type=Path, required=True)
    parser.add_argument("--attention-metadata", type=Path, required=True)
    parser.add_argument("--memory-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    attention_metadata = read_json(args.attention_metadata)
    correctness = read_json(args.correctness_json)
    all_rows: list[dict[str, str]] = []
    for path in (
        args.checkpoint_csv,
        args.baseline_csv,
        args.compile_csv,
        args.flash_csv,
    ):
        all_rows.extend(csv_rows(path))

    allocated = [float(row["peak_allocated_mib"]) for row in all_rows if row.get("peak_allocated_mib") not in ("", None)]
    reserved = [float(row["peak_reserved_mib"]) for row in all_rows if row.get("peak_reserved_mib") not in ("", None)]
    peak_allocated = max(allocated)
    peak_reserved = max(reserved)
    within_budget = peak_reserved <= ALLOCATOR_LIMIT_MIB and peak_allocated <= ALLOCATOR_LIMIT_MIB
    memory = {
        "allocator": {
            "allocator_fraction": attention_metadata["allocator"]["allocator_fraction"],
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": peak_allocated,
        "pytorch_peak_reserved_mib": peak_reserved,
        "within_24gib": within_budget,
        "scope": ("maximum across checkpoint, attention baseline, compile, and Flash benchmark processes"),
    }
    write_json(args.memory_output, memory)

    gpu = attention_metadata["gpu"]
    metadata = {
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "seed": 2026,
        "hardware": gpu,
        "hardware_disclosure": ("The physical RTX 4090 reports 49140 MiB; every formal process was constrained to the assignment's stricter 23552 MiB PyTorch allocator budget."),
        "allocator": attention_metadata["allocator"],
        "software": {
            "driver": gpu["driver_version"],
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "triton": triton.__version__,
        },
        "power_limit_w": gpu["power_limit_w"],
        "pstate_at_attention_start": gpu["pstate"],
        "tf32": {
            "performance": True,
            "fp32_correctness": False,
        },
        "compile": {
            "backend": "inductor",
            "cold_start_separated": True,
            "shape_specialization": True,
            "cold_start_cache": ("fresh process and empty TORCHINDUCTOR_CACHE_DIR per workload"),
        },
        "timing": {
            "attention_timer": "triton.testing.do_bench",
            "attention_warmup_ms": 100,
            "attention_rep_ms": 300,
            "attention_quantiles": [0.2, 0.5, 0.8],
            "checkpoint_warmup_steps": 3,
            "checkpoint_measurement_steps": 5,
        },
        "correctness_summary": correctness["summary"],
        "commands": [
            "python -m student_scripts.a2k.run_checkpointing --output results/checkpointing.csv",
            "python -m student_scripts.a2k.run_compile_comparison --output results/compile_comparison.csv",
            "python -m student_scripts.a2k.run_correctness --output results/correctness.json",
            "python -m student_scripts.a2k.run_attention_benchmarks --flash-output results/flash_benchmark.csv --baseline-output results/attention_baseline.csv",
            "python -m student_scripts.a2k.run_unit_tests --output results/unit_tests.txt",
        ],
        "process_policy": (
            "checkpoint, compile, correctness, unit tests, and attention matrix were executed serially in fresh Python processes with CUDA_VISIBLE_DEVICES exposing one GPU"
        ),
    }
    write_json(args.metadata_output, metadata)
    return 0 if within_budget else 1


if __name__ == "__main__":
    raise SystemExit(main())
