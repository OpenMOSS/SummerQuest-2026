"""Create lightweight CSV summaries from A2-P JSONL result records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {error}") from error
    return records


def write_benchmark_csv(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_utc",
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "warmup_steps",
        "measurement_steps",
        "mean_ms",
        "std_ms",
        "cv",
        "raw_timings_ms",
        "parameter_count",
        "device_name",
        "torch_version",
        "cuda_version",
        "active_bytes",
        "peak_active_bytes",
        "allocated_bytes",
        "peak_allocated_bytes",
        "reserved_bytes",
        "peak_reserved_bytes",
        "command",
    ]
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            model = record["model_config"]
            run = record["run_config"]
            stats = record["statistics"]
            environment = record["environment"]
            memory = record.get("memory") or {}
            memory_stats = memory.get("statistics_bytes") or {}
            writer.writerow(
                {
                    "timestamp_utc": record["timestamp_utc"],
                    "model_size": run["model_size"],
                    "mode": run["mode"],
                    "dtype": run["precision"],
                    "batch_size": model["batch_size"],
                    "context_length": model["context_length"],
                    "warmup_steps": run["warmup_steps"],
                    "measurement_steps": run["measurement_steps"],
                    "mean_ms": stats["mean_ms"],
                    "std_ms": stats["std_ms"],
                    "cv": stats["cv"],
                    "raw_timings_ms": json.dumps(record["raw_timings_ms"]),
                    "parameter_count": record["parameter_count"],
                    "device_name": environment["device_name"],
                    "torch_version": environment["torch_version"],
                    "cuda_version": environment["cuda_version"],
                    "active_bytes": memory_stats.get("active_bytes"),
                    "peak_active_bytes": memory_stats.get("peak_active_bytes"),
                    "allocated_bytes": memory_stats.get("allocated_bytes"),
                    "peak_allocated_bytes": memory_stats.get("peak_allocated_bytes"),
                    "reserved_bytes": memory_stats.get("reserved_bytes"),
                    "peak_reserved_bytes": memory_stats.get("peak_reserved_bytes"),
                    "command": record["command"],
                }
            )


def write_memory_csv(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "active_bytes",
        "peak_active_bytes",
        "allocated_bytes",
        "peak_allocated_bytes",
        "reserved_bytes",
        "peak_reserved_bytes",
        "snapshot_file",
    ]
    with output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            memory = record.get("memory")
            if memory is None or memory.get("statistics_bytes") is None:
                continue
            model = record["model_config"]
            run = record["run_config"]
            writer.writerow(
                {
                    "model_size": run["model_size"],
                    "mode": run["mode"],
                    "dtype": run["precision"],
                    "batch_size": model["batch_size"],
                    "context_length": model["context_length"],
                    **memory["statistics_bytes"],
                    "snapshot_file": memory["snapshot_file"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSONL into submission-safe CSV files.")
    parser.add_argument("kind", choices=("benchmark", "memory"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = read_jsonl(args.input)
    if args.kind == "benchmark":
        write_benchmark_csv(records, args.output)
    else:
        write_memory_csv(records, args.output)


if __name__ == "__main__":
    main()
