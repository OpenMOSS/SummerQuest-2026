from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    for name in (
        "checkpointing.csv",
        "attention_baseline.csv",
        "compile_comparison.csv",
        "flash_benchmark.csv",
    ):
        path = args.results / name
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            statuses = sorted({row.get("status", "") for row in rows})
            print(f"{name}: rows={len(rows)} statuses={statuses}")


if __name__ == "__main__":
    main()
