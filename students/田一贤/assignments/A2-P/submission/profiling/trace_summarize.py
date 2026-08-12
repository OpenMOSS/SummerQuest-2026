"""Create lightweight, structured summaries from a raw Chrome trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--gpu", default="NVIDIA H200")
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    payload = json.loads(raw)
    events = payload.get("traceEvents", payload)
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "cpu_time_us": 0.0, "cuda_time_us": 0.0}
    )
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category not in {"cpu_op", "kernel"}:
            continue
        name = str(event.get("name", ""))
        duration = float(event.get("dur", 0.0) or 0.0)
        totals[name]["calls"] += 1
        if category == "cpu_op":
            totals[name]["cpu_time_us"] += duration
        else:
            totals[name]["cuda_time_us"] += duration
    rows = [
        {
            "name": name,
            "calls": int(values["calls"]),
            "cpu_time_us": values["cpu_time_us"],
            "cuda_time_us": values["cuda_time_us"],
            "status": "measured_from_raw_trace",
        }
        for name, values in totals.items()
    ]
    rows.sort(
        key=lambda row: float(row["cpu_time_us"]) + float(row["cuda_time_us"]),
        reverse=True,
    )
    rows = rows[:30]

    def stage_events(stage: str, category: str) -> list[dict]:
        return [
            event
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") == category
            and event.get("name") == stage
        ]

    kernels = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ]
    stage_rows = []
    for stage in ("forward", "backward", "optimizer"):
        cpu_events = stage_events(stage, "user_annotation")
        cuda_events = stage_events(stage, "gpu_user_annotation")
        kernel_events = []
        for cpu_event in cpu_events:
            start = float(cpu_event.get("ts", 0.0))
            end = start + float(cpu_event.get("dur", 0.0))
            kernel_events.extend(
                event
                for event in kernels
                if start <= float(event.get("ts", -1.0)) <= end
            )
        stage_rows.append(
            {
                "stage": stage,
                "cpu_calls": len(cpu_events),
                "cpu_total_us": sum(float(event.get("dur", 0.0)) for event in cpu_events),
                "cuda_annotation_calls": len(cuda_events),
                "cuda_annotation_total_us": sum(
                    float(event.get("dur", 0.0)) for event in cuda_events
                ),
                "cuda_kernel_calls": len(kernel_events),
                "cuda_kernel_total_us": sum(
                    float(event.get("dur", 0.0)) for event in kernel_events
                ),
                "status": "measured_from_raw_trace" if cpu_events else "missing_phase_event",
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trace_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "stage_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    metadata = {
        "status": "pass"
        if all(row["status"] == "measured_from_raw_trace" for row in stage_rows)
        else "incomplete",
        "measurement_collected": True,
        "evaluation_type": "self_supervised_proxy",
        "tool": "torch.profiler Chrome trace",
        "gpu": args.gpu,
        "model_size": "small",
        "context_length": 512,
        "batch_size": 4,
        "dtype": args.dtype,
        "profiled_steps": 1,
        "warmup_forward_steps": 1,
        "stage_ranges": ["forward", "backward", "optimizer"],
        "raw_trace": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "submitted": False,
            "retention": "remote execution workspace only",
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
