#!/usr/bin/env python3
"""Summarize one or more JSONL training logs for tables and reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from _common import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize train/validation metrics from JSONL logs.")
    parser.add_argument("metrics", type=Path, nargs="+")
    parser.add_argument("--label", action="append", help="repeat once per metrics file")
    parser.add_argument("--output", type=Path, help="optional JSON destination")
    return parser.parse_args()


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(event, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            events.append(event)
    return events


def summarize(path: Path, label: str) -> dict[str, Any]:
    events = load_events(path)
    run_start = next((event for event in events if event.get("event") == "run_start"), {})

    def finite_float(value: Any) -> float | None:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    train = [event for event in events if event.get("event") == "train" and finite_float(event.get("loss")) is not None]
    validation = [
        event for event in events if event.get("event") == "validation" and finite_float(event.get("loss")) is not None
    ]
    terminal = next(
        (event for event in reversed(events) if event.get("event") in {"run_end", "diverged"}),
        events[-1] if events else {},
    )
    best_validation = min(validation, key=lambda event: float(event["loss"])) if validation else None
    throughputs = []
    for event in train:
        throughput = finite_float(event.get("tokens_per_second"))
        if throughput is not None and throughput > 0:
            throughputs.append(throughput)

    def maximum_integer(field: str) -> int | None:
        values: list[int] = []
        for event in events:
            value = finite_float(event.get(field))
            if value is not None:
                values.append(int(value))
        return max(values) if values else None

    return {
        "label": label,
        "metrics_file": path.name,
        "run_name": run_start.get("run_name"),
        "device_type": run_start.get("device_type"),
        "precision": run_start.get("precision"),
        "parameter_count": run_start.get("parameter_count"),
        "effective_batch_size": run_start.get("effective_batch_size"),
        "status": terminal.get("status", terminal.get("event", "running")),
        "final_step": int(terminal.get("step", train[-1]["step"] if train else 0)),
        "processed_tokens": int(terminal.get("processed_tokens", train[-1]["processed_tokens"] if train else 0)),
        "wall_time_seconds": finite_float(terminal.get("wall_time_seconds")),
        "final_train_loss": float(train[-1]["loss"]) if train else None,
        "final_validation_loss": float(validation[-1]["loss"]) if validation else None,
        "best_validation_loss": float(best_validation["loss"]) if best_validation else None,
        "best_validation_step": int(best_validation["step"]) if best_validation else None,
        "median_train_tokens_per_second": statistics.median(throughputs) if throughputs else None,
        "peak_cuda_memory_allocated_bytes": maximum_integer("cuda_peak_memory_allocated_bytes"),
        "peak_cuda_memory_reserved_bytes": maximum_integer("cuda_peak_memory_reserved_bytes"),
        "train_points": len(train),
        "validation_points": len(validation),
    }


def main() -> None:
    args = parse_args()
    if args.label is not None and len(args.label) != len(args.metrics):
        raise ValueError("repeat --label exactly once per metrics file")
    labels = args.label or [path.parent.name or path.stem for path in args.metrics]
    payload = {
        "format": "cs336-metrics-summary-v1",
        "runs": [summarize(path, label) for path, label in zip(args.metrics, labels, strict=True)],
    }
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
