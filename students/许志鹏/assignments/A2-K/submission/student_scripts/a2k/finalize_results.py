from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from cs336_systems.a2k.runtime import HARD_GPU_LIMIT_MIB, DEFAULT_ALLOCATOR_LIMIT_MIB, write_json


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Build the lightweight A2-K metadata and memory evidence files")
    result.add_argument("--results-dir", type=Path, default=Path("results"))
    result.add_argument("--commit", default=None)
    return result.parse_args()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parser()
    results_dir = args.results_dir
    metadata_records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_metadata.json")):
        if path.name == "run_metadata.json":
            continue
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            metadata_records.extend(item for item in loaded if isinstance(item, dict))
        elif isinstance(loaded, dict):
            metadata_records.append(loaded)

    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob("*.csv")):
        rows.extend(_rows(path))
    allocated = [_number(row.get("peak_allocated_mib")) for row in rows]
    reserved = [_number(row.get("peak_reserved_mib")) for row in rows]
    allocated = [value for value in allocated if value is not None]
    reserved = [value for value in reserved if value is not None]
    allocator_fraction = next(
        (
            float(record["allocator"]["allocator_fraction"])
            for record in metadata_records
            if isinstance(record.get("allocator"), dict) and "allocator_fraction" in record["allocator"]
        ),
        0.0,
    )
    hardware = next((record.get("gpu") for record in metadata_records if isinstance(record.get("gpu"), dict)), {})
    commit = args.commit or _commit()

    write_json(
        results_dir / "run_metadata.json",
        {
            "commit": commit,
            "runs": metadata_records,
            "gpu": hardware,
            "allocator": {"allocator_limit_mib": DEFAULT_ALLOCATOR_LIMIT_MIB, "allocator_fraction": allocator_fraction},
        },
    )
    peak_allocated = max(allocated, default=0.0)
    peak_reserved = max(reserved, default=0.0)
    write_json(
        results_dir / "memory_evidence.json",
        {
            "allocator": {"allocator_fraction": allocator_fraction, "allocator_limit_mib": DEFAULT_ALLOCATOR_LIMIT_MIB},
            "hard_limit_mib": HARD_GPU_LIMIT_MIB,
            "pytorch_peak_allocated_mib": peak_allocated,
            "pytorch_peak_reserved_mib": peak_reserved,
            "within_24gib": peak_reserved <= DEFAULT_ALLOCATOR_LIMIT_MIB and peak_reserved <= HARD_GPU_LIMIT_MIB,
            "source_files": sorted(path.name for path in results_dir.glob("*.csv")),
        },
    )
    print(f"wrote {results_dir / 'run_metadata.json'}")
    print(f"wrote {results_dir / 'memory_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
