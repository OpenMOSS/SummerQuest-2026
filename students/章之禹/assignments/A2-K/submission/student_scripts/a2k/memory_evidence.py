"""Aggregate public peak-memory evidence from isolated A2-K result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    for path in sorted(args.results.rglob("*.json")):
        if path.name in {"memory_evidence.json", "run_metadata.json"}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if "peak_allocated_mib" in payload or "peak_reserved_mib" in payload:
            records.append(
                {
                    "source": path.name,
                    "peak_allocated_mib": float(payload.get("peak_allocated_mib", 0.0)),
                    "peak_reserved_mib": float(payload.get("peak_reserved_mib", 0.0)),
                    "status": payload.get("status"),
                }
            )
        for key in ("eager", "compiled"):
            nested = payload.get(key)
            if isinstance(nested, dict) and "peak_allocated_mib" in nested:
                records.append(
                    {
                        "source": f"{path.name}:{key}",
                        "peak_allocated_mib": float(nested.get("peak_allocated_mib", 0.0)),
                        "peak_reserved_mib": float(nested.get("peak_reserved_mib", 0.0)),
                        "status": payload.get("status"),
                    }
                )
    max_allocated = max((row["peak_allocated_mib"] for row in records), default=0.0)
    max_reserved = max((row["peak_reserved_mib"] for row in records), default=0.0)
    first_metadata = next(
        (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.results.rglob("*.json"))
            if path.name not in {"memory_evidence.json", "run_metadata.json"}
            and path.is_file()
        ),
        {},
    )
    allocator = first_metadata.get("allocator", {}) if isinstance(first_metadata, dict) else {}
    payload = {
        "allocator": {
            "allocator_fraction": float(allocator.get("allocator_fraction", 1.0)),
            "allocator_limit_mib": 23552,
        },
        "hard_limit_mib": 24576,
        "pytorch_peak_allocated_mib": max_allocated,
        "pytorch_peak_reserved_mib": max_reserved,
        "within_24gib": max_reserved <= 23552 and max_allocated <= 24576,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
