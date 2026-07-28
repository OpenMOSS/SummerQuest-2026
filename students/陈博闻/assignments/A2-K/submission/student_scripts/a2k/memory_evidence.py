from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from student_scripts.a2k.common import ALLOCATOR_LIMIT_MIB, write_json


def read_peaks(paths: list[Path]) -> tuple[float, float]:
    peak_allocated = 0.0
    peak_reserved = 0.0
    for path in paths:
        if not path.exists():
            continue
        with path.open() as f:
            for row in csv.DictReader(f):
                try:
                    peak_allocated = max(peak_allocated, float(row.get("peak_allocated_mib") or 0.0))
                    peak_reserved = max(peak_reserved, float(row.get("peak_reserved_mib") or 0.0))
                except ValueError:
                    continue
    return peak_allocated, peak_reserved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/memory_evidence.json"))
    args = parser.parse_args()
    peak_allocated, peak_reserved = read_peaks(
        [
            args.results_dir / "checkpointing.csv",
            args.results_dir / "attention_baseline.csv",
            args.results_dir / "compile_comparison.csv",
            args.results_dir / "flash_benchmark.csv",
        ]
    )
    metadata_path = args.results_dir / "run_metadata.json"
    allocator_fraction = 1.0
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
            allocator_fraction = float(metadata.get("allocator_fraction", allocator_fraction))
        except (TypeError, ValueError, json.JSONDecodeError):
            allocator_fraction = 1.0
    payload = {
        "allocator": {
            "allocator_fraction": allocator_fraction,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": 24 * 1024,
        "pytorch_peak_allocated_mib": peak_allocated,
        "pytorch_peak_reserved_mib": peak_reserved,
        "within_24gib": peak_reserved <= ALLOCATOR_LIMIT_MIB,
    }
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
