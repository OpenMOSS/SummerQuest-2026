"""Aggregate per-configuration memory outputs into public A2-P summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-shard", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = []
    for shard in args.memory_shard:
        peaks = shard / "peaks.csv"
        with peaks.open(encoding="utf-8") as f:
            shard_rows = list(csv.DictReader(f))
        rows.extend(shard_rows)
        sources.append({"file": peaks.name, "sha256": _sha256(peaks)})
    key = lambda row: (int(row["context_length"]), row["dtype"], row["mode"])
    rows.sort(key=key)
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError("duplicate memory configurations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    peaks_out = args.output_dir / "peaks.csv"
    with peaks_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    timeline_hashes = {
        path.name: _sha256(path)
        for path in sorted(args.output_dir.glob("memory_*_timeline_*.csv"))
    }
    metadata = {
        "status": "formal_run_complete"
        if rows and all(row.get("status") == "pass" for row in rows)
        else "incomplete",
        "measurement_collected": bool(rows),
        "history_started_after_warmup": True,
        "evaluation_type": "self_supervised_proxy",
        "configurations": len(rows),
        "configuration_keys": [key(row) for row in rows],
        "sources": sources,
        "peaks_sha256": _sha256(peaks_out),
        "timeline_sha256": timeline_hashes,
        "raw_snapshots_submitted": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
