from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_benchmark(source: Path, destination: Path) -> None:
    rows = load_jsonl(source)
    deduplicated: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("model_size"),
            row.get("batch_size"),
            row.get("context_length"),
            row.get("mode"),
            row.get("dtype"),
            row.get("warmup_steps"),
            row.get("measurement_steps"),
            row.get("seed"),
        )
        deduplicated.setdefault(key, row)
    rows = list(deduplicated.values())
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "status",
        "model_size",
        "batch_size",
        "context_length",
        "mode",
        "dtype",
        "seed",
        "warmup_steps",
        "measurement_steps",
        "raw_ms",
        "mean_ms",
        "sample_std_ms",
        "cv",
        "last_loss",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "command",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if isinstance(row.get("raw_ms"), list):
                row["raw_ms"] = json.dumps(row["raw_ms"])
            writer.writerow(row)


def _device_events_for_range(events: list[dict], start: float, end: float) -> list[dict]:
    correlations = {
        event.get("args", {}).get("correlation")
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"} and event.get("ph") == "X" and start <= float(event.get("ts", -1)) <= end
    }
    correlations.discard(None)
    return [
        event
        for event in events
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"} and event.get("ph") == "X" and event.get("args", {}).get("correlation") in correlations
    ]


def trace_stage_rows(trace_path: Path, run: str) -> list[dict]:
    payload = json.loads(trace_path.read_text())
    events = payload.get("traceEvents", [])
    wanted = {
        "profile/measure",
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    }
    ranges: dict[str, list[dict]] = {name: [] for name in wanted}
    for event in events:
        if event.get("cat") == "user_annotation" and event.get("ph") == "X" and event.get("name") in wanted:
            ranges[event["name"]].append(event)

    rows = []
    for name in sorted(wanted):
        annotations = ranges[name]
        if not annotations:
            continue
        device_events = []
        cpu_durations = []
        for annotation in annotations:
            start = float(annotation["ts"])
            end = start + float(annotation.get("dur", 0.0))
            cpu_durations.append(float(annotation.get("dur", 0.0)) / 1000)
            device_events.extend(_device_events_for_range(events, start, end))
        unique_device_events = {
            (
                event.get("cat"),
                float(event.get("ts", 0.0)),
                float(event.get("dur", 0.0)),
                event.get("name"),
            ): event
            for event in device_events
        }
        device_events = list(unique_device_events.values())
        device_sum_ms = sum(float(event.get("dur", 0.0)) for event in device_events) / 1000
        if device_events:
            first = min(float(event["ts"]) for event in device_events)
            last = max(float(event["ts"]) + float(event.get("dur", 0.0)) for event in device_events)
            device_elapsed_ms = (last - first) / 1000
        else:
            device_elapsed_ms = 0.0
        rows.append(
            {
                "run": run,
                "stage": name,
                "calls": len(annotations),
                "cpu_wall_sum_ms": sum(cpu_durations),
                "cpu_wall_median_ms": statistics.median(cpu_durations),
                "cuda_device_event_count": len(device_events),
                "cuda_device_time_sum_ms": device_sum_ms,
                "cuda_device_elapsed_ms": device_elapsed_ms,
            }
        )
    return rows


def trace_kernel_rows(trace_path: Path, run: str, limit: int = 12) -> list[dict]:
    payload = json.loads(trace_path.read_text())
    aggregates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in payload.get("traceEvents", []):
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        values = aggregates[event.get("name", "unknown kernel")]
        values[0] += 1
        values[1] += float(event.get("dur", 0.0))
    rows = []
    for name, (calls, cuda_time_us) in sorted(aggregates.items(), key=lambda item: item[1][1], reverse=True)[:limit]:
        rows.append(
            {
                "run": run,
                "model_size": run.split("_ctx", 1)[0],
                "context_length": run.split("_ctx", 1)[1].split("_", 1)[0],
                "dtype": run.rsplit("_", 1)[-1],
                "entry_type": "cuda_kernel",
                "op_or_range": name,
                "calls": int(calls),
                "cpu_time_total_us": 0.0,
                "cuda_time_total_us": cuda_time_us,
                "self_cpu_time_total_us": 0.0,
                "self_cuda_time_total_us": cuda_time_us,
            }
        )
    return rows


def trace_stage_kernel_rows(trace_path: Path, run: str, limit: int = 8) -> list[dict]:
    payload = json.loads(trace_path.read_text())
    events = payload.get("traceEvents", [])
    rows = []
    for stage in ("forward", "backward", "optimizer"):
        annotations = [event for event in events if event.get("cat") == "user_annotation" and event.get("ph") == "X" and event.get("name") == stage]
        stage_events = []
        for annotation in annotations:
            start = float(annotation["ts"])
            end = start + float(annotation.get("dur", 0.0))
            stage_events.extend(_device_events_for_range(events, start, end))
        aggregates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for event in stage_events:
            if event.get("cat") != "kernel":
                continue
            values = aggregates[event.get("name", "unknown kernel")]
            values[0] += 1
            values[1] += float(event.get("dur", 0.0))
        for name, (calls, cuda_time_us) in sorted(aggregates.items(), key=lambda item: item[1][1], reverse=True)[:limit]:
            rows.append(
                {
                    "run": run,
                    "stage": stage,
                    "kernel": name,
                    "calls": int(calls),
                    "cuda_time_total_us": cuda_time_us,
                }
            )
    return rows


def combine_profile(
    source_dir: Path,
    destination: Path,
    stage_destination: Path,
    metadata_destination: Path,
) -> None:
    rows = []
    stage_rows = []
    stage_kernel_rows = []
    metadata = []
    for path in sorted(source_dir.glob("*.csv")):
        with path.open() as handle:
            run_rows = list(csv.DictReader(handle))
        for row in run_rows:
            row["entry_type"] = "op_or_range"
        selected_ranges = {
            "profile/warmup",
            "profile/measure",
            "forward",
            "backward",
            "optimizer",
            "attention/scores",
            "attention/softmax",
            "attention/value",
        }
        top_cuda = sorted(run_rows, key=lambda row: float(row["cuda_time_total_us"]), reverse=True)[:12]
        rows.extend([row for row in run_rows if row["op_or_range"] in selected_ranges])
        rows.extend(row for row in top_cuda if row not in rows)
        trace_path = path.with_suffix(".trace.json")
        if trace_path.is_file():
            stage_rows.extend(trace_stage_rows(trace_path, path.stem))
            rows.extend(trace_kernel_rows(trace_path, path.stem))
            stage_kernel_rows.extend(trace_stage_kernel_rows(trace_path, path.stem))
    for path in sorted(source_dir.glob("*.metadata.json")):
        metadata.append(json.loads(path.read_text()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if stage_rows:
        with stage_destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(stage_rows)
    if stage_kernel_rows:
        with (stage_destination.parent / "stage_kernel_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(stage_kernel_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(stage_kernel_rows)
    metadata_destination.write_text(json.dumps(metadata, indent=2) + "\n")


def combine_memory(source_dir: Path, destination: Path, metadata_destination: Path) -> None:
    metadata = [json.loads(path.read_text()) for path in sorted(source_dir.glob("*.metadata.json"))]
    for row in metadata:
        largest = row.get("largest_allocation_mib")
        peak = row.get("peak_allocated_mib")
        if largest is not None and peak is not None and float(largest) > float(peak):
            row["largest_allocation_mib"] = None
            row["largest_allocation_note"] = "unavailable: legacy aggregate exceeded the measured peak"
    fields = [
        "status",
        "model_size",
        "batch_size",
        "context_length",
        "mode",
        "dtype",
        "peak_active_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "largest_allocation_mib",
        "local_timeline_file",
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(metadata)
    metadata_destination.write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight A2-P public result files.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_benchmark(args.raw_root / "benchmark.jsonl", args.public_root / "benchmark.csv")
    shutil.copy2(
        args.raw_root / "mixed_precision.json",
        args.public_root / "mixed_precision.json",
    )
    combine_profile(
        args.raw_root / "profile",
        args.public_root / "profile" / "trace_summary.csv",
        args.public_root / "profile" / "stage_summary.csv",
        args.public_root / "profile" / "run_metadata.json",
    )
    combine_memory(
        args.raw_root / "memory",
        args.public_root / "memory" / "peaks.csv",
        args.public_root / "memory" / "run_metadata.json",
    )


if __name__ == "__main__":
    main()
