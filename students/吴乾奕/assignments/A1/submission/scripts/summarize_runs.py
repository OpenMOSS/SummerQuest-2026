#!/usr/bin/env python3
"""Summarize JSONL training logs as CSV and a Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


FIELDS = [
    "run_name",
    "status",
    "final_step",
    "processed_tokens",
    "best_validation_loss",
    "final_train_loss",
    "elapsed_seconds",
    "peak_tokens_per_second",
    "peak_cuda_memory_bytes",
    "log_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, action="append", type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


def summarize(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty log: {path}")
    validation_losses = [float(record["validation_loss"]) for record in records if "validation_loss" in record]
    train_records = [record for record in records if record.get("event") == "train"]
    last_record = records[-1]
    terminal_events = {"completed", "divergent", "interrupted", "oom", "failed"}
    status = last_record.get("event") if last_record.get("event") in terminal_events else "running"
    return {
        "run_name": next((record.get("run_name") for record in records if record.get("run_name")), path.parent.name),
        "status": status,
        "final_step": max((int(record.get("step", 0)) for record in records), default=0),
        "processed_tokens": max((int(record.get("processed_tokens", 0)) for record in records), default=0),
        "best_validation_loss": min(validation_losses) if validation_losses else math.nan,
        "final_train_loss": float(train_records[-1]["train_loss"]) if train_records else math.nan,
        "elapsed_seconds": max((float(record.get("elapsed_seconds", 0)) for record in records), default=0),
        "peak_tokens_per_second": max(
            (float(record.get("step_tokens_per_second", record.get("tokens_per_second", 0))) for record in records),
            default=0,
        ),
        "peak_cuda_memory_bytes": max(
            (int(record.get("peak_cuda_memory_bytes", 0)) for record in records),
            default=0,
        ),
        "log_path": str(path),
    }


def display(value: object) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def main() -> None:
    args = parse_args()
    summaries = [summarize(path) for path in args.log]
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(FIELDS) + " |",
        "| " + " | ".join("---" for _ in FIELDS) + " |",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(display(summary[field]).replace("|", "\\|") for field in FIELDS) + " |")
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.csv_output.resolve())
    print(args.markdown_output.resolve())


if __name__ == "__main__":
    main()
