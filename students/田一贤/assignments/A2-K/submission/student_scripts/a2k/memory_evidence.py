from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .common import ALLOCATOR_LIMIT_MIB, configure_allocator_guard


def _max_column(results: Path, column: str) -> float:
    values = []
    for name in (
        "checkpointing.csv",
        "attention_baseline.csv",
        "compile_comparison.csv",
        "flash_benchmark.csv",
        "flash_boundary.csv",
    ):
        path = results / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    values.append(float(row.get(column, "")))
                except (TypeError, ValueError):
                    pass
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/memory_evidence.json")
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    runtime_guard = configure_allocator_guard()
    peak_allocated = _max_column(args.results, "peak_allocated_mib")
    peak_reserved = _max_column(args.results, "peak_reserved_mib")
    formal_files = ("checkpointing.csv", "attention_baseline.csv", "compile_comparison.csv", "flash_benchmark.csv", "flash_boundary.csv")
    guard_rows, total_rows = 0, 0
    fractions = []
    for name in formal_files:
        path = args.results / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "pass":
                    continue
                total_rows += 1
                if row.get("allocator_guard_applied", "").lower() == "true":
                    guard_rows += 1
                try:
                    fractions.append(float(row.get("allocator_fraction", "")))
                except ValueError:
                    pass
    status = "pass" if total_rows > 0 and guard_rows == total_rows else "guard_evidence_incomplete"
    data = {
        "status": status,
        "measurement_collected": peak_reserved > 0,
        "evaluation_type": "self_supervised_proxy",
        "proxy_source": "torch.cuda.max_memory_reserved",
        "allocator": {
            "runtime_guard": "torch.cuda.set_per_process_memory_fraction",
            "runtime_guard_applied": runtime_guard["applied"],
            "allocator_fraction": runtime_guard["fraction"],
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
            "guarded_pass_rows": guard_rows,
            "total_pass_rows": total_rows,
            "all_formal_pass_rows_guarded": total_rows > 0 and guard_rows == total_rows,
            "observed_allocator_fractions": sorted(set(fractions)),
        },
        "hard_limit_mib": 24576,
        "pytorch_peak_allocated_mib": peak_allocated,
        "pytorch_peak_reserved_mib": peak_reserved,
        "within_24gib": peak_reserved <= ALLOCATOR_LIMIT_MIB,
        "nvidia_smi": {
            "max_gpu_memory_used_mib": peak_reserved,
            "source": "pytorch_peak_reserved_proxy; nvidia-smi not collected",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
