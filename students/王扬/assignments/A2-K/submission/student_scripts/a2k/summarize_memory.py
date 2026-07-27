from __future__ import annotations

import argparse
import csv
from pathlib import Path

from student_scripts.a2k.common import add_common_args, ensure_dirs, write_json


def read_peak(path: Path) -> tuple[float, float]:
    if not path.exists():
        return 0.0, 0.0
    max_allocated = 0.0
    max_reserved = 0.0
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_allocated = max(max_allocated, float(row.get("peak_allocated_mib") or 0.0))
                max_reserved = max(max_reserved, float(row.get("peak_reserved_mib") or 0.0))
            except ValueError:
                continue
    return max_allocated, max_reserved


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--allocator-fraction", type=float, default=0.0)
    args = parser.parse_args()
    ensure_dirs()

    files = [
        args.output_dir / "checkpointing.csv",
        args.output_dir / "attention_baseline.csv",
        args.output_dir / "compile_comparison.csv",
        args.output_dir / "flash_benchmark.csv",
    ]
    allocated = []
    reserved = []
    for path in files:
        a, r = read_peak(path)
        allocated.append(a)
        reserved.append(r)
    peak_allocated = max(allocated, default=0.0)
    peak_reserved = max(reserved, default=0.0)
    write_json(
        args.output_dir / "memory_evidence.json",
        {
            "allocator": {
                "allocator_fraction": args.allocator_fraction,
                "allocator_limit_mib": 23552,
            },
            "hard_limit_mib": 24576,
            "pytorch_peak_allocated_mib": peak_allocated,
            "pytorch_peak_reserved_mib": peak_reserved,
            "within_24gib": peak_reserved <= 23552,
            "covered_files": [str(path) for path in files if path.exists()],
        },
    )


if __name__ == "__main__":
    main()
