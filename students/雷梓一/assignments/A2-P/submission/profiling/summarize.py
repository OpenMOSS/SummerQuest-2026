from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .common import read_json, write_json


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def benchmark_csv(inputs: list[Path], output: Path) -> None:
    rows = []
    for path in inputs:
        item = read_json(path)
        config = item["config"]
        rows.append(
            {
                **config,
                "gpu": item["software"].get("gpu_name"),
                "gpu_memory_bytes": item["software"].get("gpu_memory_bytes"),
                "python": item["software"].get("python"),
                "pytorch": item["software"].get("pytorch"),
                "cuda_runtime": item["software"].get("cuda_runtime"),
                "cudnn": item["software"].get("cudnn"),
                "raw_seconds": json.dumps(item["raw_seconds"]),
                "mean_seconds": item["mean_seconds"],
                "sample_std_seconds": item["sample_std_seconds"],
                "cv": item["cv"],
                "peak_allocated_bytes": item["memory"].get("peak_allocated_bytes"),
                "peak_reserved_bytes": item["memory"].get("peak_reserved_bytes"),
                "command": item["command"],
                "result_file": item.get("result_file", path.name),
                "driver": item["software"].get("driver"),
            }
        )
    fields = [
        "model_size", "d_model", "d_ff", "num_layers", "num_heads", "batch_size", "context_length",
        "vocab_size", "mode", "warmup", "steps", "dtype", "seed", "learning_rate", "gpu",
        "gpu_memory_bytes", "python", "pytorch", "cuda_runtime", "cudnn",
        "raw_seconds", "mean_seconds", "sample_std_seconds", "cv", "peak_allocated_bytes",
        "peak_reserved_bytes", "command", "result_file", "driver",
    ]
    write_csv(output, rows, fields)


def profile_outputs(inputs: list[Path], csv_output: Path, metadata_output: Path) -> None:
    rows = []
    runs = []
    for path in inputs:
        item = read_json(path)
        config = item["config"]
        runs.append({
            key: item[key]
            for key in (
                "timestamp_utc", "tool", "command", "trace_file", "summary_file",
                "config", "software", "stage_cuda_ms",
            )
            if key in item
        })
        for stage, cuda_ms in item["stage_cuda_ms"].items():
            rows.append(
                {
                    **config,
                    "tool": "cuda_event",
                    "name": stage,
                    "calls": 1,
                    "cpu_total_us": None,
                    "cpu_self_us": None,
                    "cuda_total_us": cuda_ms * 1000,
                    "cuda_self_us": None,
                }
            )
        for event in item["events"]:
            rows.append({**config, "tool": item["tool"], **event})
    fields = [
        "model_size", "batch_size", "context_length", "dtype", "tool", "name", "calls",
        "cpu_total_us", "cpu_self_us", "cuda_total_us", "cuda_self_us",
    ]
    write_csv(csv_output, rows, fields)
    write_json(metadata_output, {"schema_version": 1, "runs": runs})


def memory_outputs(inputs: list[Path], csv_output: Path, metadata_output: Path) -> None:
    rows = []
    runs = []
    for path in inputs:
        item = read_json(path)
        config = item["config"]
        memory = item.get("memory", {})
        largest = item.get("largest_active_allocation") or {}
        rows.append(
            {
                **config,
                "status": item["status"],
                **memory,
                "largest_active_allocation_bytes": largest.get("size_bytes"),
                "error_type": item.get("error_type"),
                "failure_stage": item.get("failure_stage"),
            }
        )
        compact = {
            key: item[key]
            for key in (
                "timestamp_utc", "command", "config", "snapshot_file", "summary_file",
                "software", "status", "memory",
            )
            if key in item
        }
        compact.update(
            {
                "error_type": item.get("error_type"),
                "error": item.get("error"),
                "failure_stage": item.get("failure_stage"),
                "largest_active_allocation": item.get("largest_active_allocation"),
                "largest_recorded_allocation": item.get("largest_recorded_allocation"),
                "transformer_blocks": item.get("transformer_blocks"),
            }
        )
        runs.append(compact)
    fields = [
        "model_size", "batch_size", "context_length", "mode", "dtype", "status", "active_bytes",
        "peak_active_bytes", "allocated_bytes", "peak_allocated_bytes", "reserved_bytes",
        "peak_reserved_bytes", "largest_active_allocation_bytes", "error_type", "failure_stage",
    ]
    write_csv(csv_output, rows, fields)
    write_json(metadata_output, {"schema_version": 1, "runs": runs})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight A2-P submission summaries")
    subparsers = parser.add_subparsers(dest="kind", required=True)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--inputs", nargs="+", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)

    profile = subparsers.add_parser("profile")
    profile.add_argument("--inputs", nargs="+", type=Path, required=True)
    profile.add_argument("--csv-output", type=Path, required=True)
    profile.add_argument("--metadata-output", type=Path, required=True)

    memory = subparsers.add_parser("memory")
    memory.add_argument("--inputs", nargs="+", type=Path, required=True)
    memory.add_argument("--csv-output", type=Path, required=True)
    memory.add_argument("--metadata-output", type=Path, required=True)

    args = parser.parse_args()
    if args.kind == "benchmark":
        benchmark_csv(args.inputs, args.output)
    elif args.kind == "profile":
        profile_outputs(args.inputs, args.csv_output, args.metadata_output)
    else:
        memory_outputs(args.inputs, args.csv_output, args.metadata_output)


if __name__ == "__main__":
    main()
