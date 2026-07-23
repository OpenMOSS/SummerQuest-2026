from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from profiling.benchmark import add_common_arguments, build_experiment, perform_warmup, run_step
from profiling.config import (
    STAGE_NAMES,
    base_metadata,
    classify_error,
    environment_metadata,
    make_run_name,
    model_config_dict,
    public_relative_path,
    safe_error_summary,
    utc_now,
    write_json,
)
from profiling.nvtx_ranges import patched_attention_ranges


EVENT_FIELDS = (
    "window",
    "stage",
    "event_type",
    "event_name",
    "calls",
    "cpu_self_us",
    "cpu_total_us",
    "cuda_self_us",
    "cuda_total_us",
    "cuda_event_elapsed_us",
    "device_type",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one stable A2-P train_step trace")
    add_common_arguments(parser)
    parser.set_defaults(mode="train_step", steps=1, warmup=5, annotate_attention=True)
    parser.add_argument(
        "--schedule-policy",
        choices=("canonical", "visible_warmup"),
        default="visible_warmup",
        help=("canonical uses profiler warmup=1, active=1; visible_warmup records the final warm-up and the unique measurement step so both user ranges are visible"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _event_name(event: Any) -> str:
    return str(getattr(event, "key", None) or getattr(event, "name", "<unnamed>"))


def _device_type(event: Any) -> str:
    value = getattr(event, "device_type", None)
    if value is None:
        return "unknown"
    text = str(value)
    return text.rsplit(".", 1)[-1].lower()


def _number(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _event_scope(event: Any) -> tuple[str, str]:
    current = event
    visited: set[int] = set()
    window = "unattributed"
    stage = "unattributed"
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        name = _event_name(current)
        if name in {"profile/warmup", "profile/measure"}:
            window = name
        elif name in STAGE_NAMES and stage == "unattributed":
            stage = name
        current = getattr(current, "cpu_parent", None)
    if stage == "unattributed" and window != "unattributed":
        stage = window
    return window, stage


def _event_type(event: Any) -> str:
    name = _event_name(event)
    if name in STAGE_NAMES:
        return "stage"
    device = _device_type(event)
    if "cuda" in device or "gpu" in device:
        return "kernel"
    return "op"


def serialize_events(events: Iterable[Any], path: Path) -> list[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            name = _event_name(event)
            event_type = _event_type(event)
            window, stage = _event_scope(event)
            row = {
                "window": window,
                "stage": name if event_type == "stage" else stage,
                "event_type": event_type,
                "event_name": name,
                "calls": int(getattr(event, "count", 1) or 1),
                "cpu_self_us": _number(event, "self_cpu_time_total", "self_cpu_time"),
                "cpu_total_us": _number(event, "cpu_time_total", "cpu_time"),
                "cuda_self_us": _number(
                    event,
                    "self_device_time_total",
                    "self_cuda_time_total",
                ),
                "cuda_total_us": _number(
                    event,
                    "device_time_total",
                    "cuda_time_total",
                ),
                "device_type": _device_type(event),
            }
            time_range = getattr(event, "time_range", None)
            if time_range is not None:
                row["start_us"] = float(getattr(time_range, "start", 0.0))
                row["end_us"] = float(getattr(time_range, "end", 0.0))
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows


def aggregate_rows(rows: Iterable[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["event_type"] != event_type:
            continue
        key = (str(row["stage"]), str(row["event_name"]))
        target = aggregated.setdefault(
            key,
            {
                "window": row.get("window", "profile/measure"),
                "stage": key[0],
                "event_type": event_type,
                "event_name": key[1],
                "calls": 0,
                "cpu_self_us": 0.0,
                "cpu_total_us": 0.0,
                "cuda_self_us": 0.0,
                "cuda_total_us": 0.0,
                "device_type": row.get("device_type", "unknown"),
            },
        )
        for field in ("calls", "cpu_self_us", "cpu_total_us", "cuda_self_us", "cuda_total_us"):
            target[field] += row[field]
    return sorted(
        aggregated.values(),
        key=lambda item: (
            item["stage"],
            -float(item["cuda_total_us"]),
            -float(item["cpu_total_us"]),
            item["event_name"],
        ),
    )


def stage_rows(rows: Iterable[dict[str, Any]], cuda_event_elapsed_us: dict[str, float] | None = None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["event_type"] != "stage":
            continue
        output_row = {field: row.get(field) for field in EVENT_FIELDS}
        output_row["cuda_event_elapsed_us"] = (cuda_event_elapsed_us or {}).get(str(row.get("stage")))
        output.append(output_row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in EVENT_FIELDS})


def _chrome_kernel_rows(trace_path: Path) -> list[dict[str, Any]]:
    """Fallback raw-kernel aggregation for traces where prof.events omits kernels."""

    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else []
    measurement_ranges = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X" or event.get("name") != "profile/measure":
            continue
        start = float(event.get("ts", 0.0) or 0.0)
        measurement_ranges.append((start, start + float(event.get("dur", 0.0) or 0.0)))

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "window": "profile/measure" if measurement_ranges else "unattributed",
            "stage": "profile/measure",
            "event_type": "kernel",
            "event_name": "",
            "calls": 0,
            "cpu_self_us": 0.0,
            "cpu_total_us": 0.0,
            "cuda_self_us": 0.0,
            "cuda_total_us": 0.0,
            "device_type": "cuda",
        }
    )
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        category = str(event.get("cat", "")).lower()
        if "kernel" not in category:
            continue
        start = float(event.get("ts", 0.0) or 0.0)
        duration = float(event.get("dur", 0.0) or 0.0)
        if measurement_ranges and not any(range_start <= start < range_end for range_start, range_end in measurement_ranges):
            continue
        name = str(event.get("name", "<unnamed-kernel>"))
        row = grouped[name]
        row["event_name"] = name
        row["calls"] += 1
        row["cuda_self_us"] += duration
        row["cuda_total_us"] += duration
    return sorted(grouped.values(), key=lambda row: -float(row["cuda_total_us"]))


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode != "train_step":
        raise ValueError("Task 2 formal traces must use --mode train_step")
    if args.steps != 1:
        raise ValueError("Task 2 captures exactly one formal measurement step")
    if args.warmup < 1:
        raise ValueError("Task 2 requires at least one warm-up step")
    if not args.device.startswith("cuda"):
        raise ValueError("Task 2 requires CUDA activities")

    run_name = make_run_name(
        f"profile-{args.run_id}",
        args.model_size,
        args.batch_size,
        args.context_length,
        args.mode,
        args.dtype,
        "torchprof",
    )
    output_dir = args.output_dir
    trace_path = output_dir / f"{run_name}.trace.json"
    events_path = output_dir / f"{run_name}.events.jsonl"
    operators_path = output_dir / f"{run_name}.operators.csv"
    kernels_path = output_dir / f"{run_name}.kernels.csv"
    stages_path = output_dir / f"{run_name}.stages.csv"
    metadata_path = output_dir / f"{run_name}.metadata.json"

    payload = base_metadata(args.run_id, run_name, "torch.profiler")
    payload.update(
        {
            "kind": "compute_profile_result",
            "model_size": args.model_size,
            "model_config": model_config_dict(args.model_size),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "dtype": args.dtype,
            "autocast_dtype": "bfloat16" if args.dtype == "bf16" else None,
            "warmup_steps": args.warmup,
            "measurement_steps": 1,
            "seed": args.seed,
            "optimizer": "cs336_basics.optimizer.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "trace_file": public_relative_path(trace_path),
            "operator_file": public_relative_path(operators_path),
            "kernel_file": public_relative_path(kernels_path),
            "stage_file": public_relative_path(stages_path),
            "event_file": public_relative_path(events_path),
            "stage_markers": list(STAGE_NAMES),
            "tool_config": {
                "activities": ["CPU", "CUDA"],
                "record_shapes": args.record_shapes,
                "profile_memory": False,
                "with_stack": False,
                "schedule_policy": args.schedule_policy,
            },
        }
    )

    failure_stage = "initialization"
    trace_written = False
    captured_rows: list[dict[str, Any]] = []
    try:
        state = build_experiment(
            model_size=args.model_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            mode=args.mode,
            dtype=args.dtype,
            seed=args.seed,
            device_name=args.device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        payload.update(
            {
                "parameter_count": state.parameter_count,
                "model_fingerprint": state.model_fingerprint,
                "environment": environment_metadata(torch),
            }
        )

        failure_stage = "warmup"
        unprofiled_warmup = max(args.warmup - 1, 0)
        attention_context = patched_attention_ranges() if args.annotate_attention else nullcontext()
        with attention_context:
            perform_warmup(state, unprofiled_warmup, label=True)

            if args.schedule_policy == "canonical":
                schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            else:
                schedule = torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=1)

            output_dir.mkdir(parents=True, exist_ok=True)

            def trace_handler(profiler: torch.profiler.profile) -> None:
                nonlocal captured_rows, trace_written
                profiler.export_chrome_trace(str(trace_path))
                trace_written = True
                captured_rows = serialize_events(profiler.events(), events_path)

            failure_stage = "profiler_capture"
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=schedule,
                on_trace_ready=trace_handler,
                record_shapes=args.record_shapes,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                run_step(state, outer_label="profile/warmup", synchronize_at_end=True)
                profiler.step()
                cuda_stage_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
                measured = run_step(
                    state,
                    outer_label="profile/measure",
                    capture_probe=True,
                    synchronize_at_end=True,
                    stage_callback=lambda stage: payload.update(failure_stage=stage),
                    cuda_stage_events=cuda_stage_events,
                )
                profiler.step()

            if not trace_written:
                profiler.export_chrome_trace(str(trace_path))
                trace_written = True

        failure_stage = "summary_export"
        serialized = captured_rows or serialize_events(profiler.events(), events_path)
        measurement_rows = [row for row in serialized if row.get("window") == "profile/measure"]
        operators = aggregate_rows(measurement_rows, "op")
        kernels = aggregate_rows(measurement_rows, "kernel")
        if not kernels:
            kernels = _chrome_kernel_rows(trace_path)
        cuda_event_elapsed_us = {stage: 1000.0 * sum(start.elapsed_time(end) for start, end in pairs) for stage, pairs in cuda_stage_events.items()}
        stages = stage_rows(serialized, cuda_event_elapsed_us)
        write_csv(operators_path, operators)
        write_csv(kernels_path, kernels)
        write_csv(stages_path, stages)

        payload.update(
            {
                "status": "success",
                "failure_stage": None,
                "last_loss": float(measured.loss.float().item()) if measured.loss is not None else None,
                "logits_finite": (bool(torch.isfinite(measured.output_probe.reshape(-1)[:4096]).all().item()) if measured.output_probe is not None else None),
                "event_count": len(serialized),
                "operator_row_count": len(operators),
                "kernel_row_count": len(kernels),
                "stage_row_count": len(stages),
                "cuda_event_stage_elapsed_us": cuda_event_elapsed_us,
                "trace_validation": {
                    "markers_seen": sorted({str(row["stage"]) for row in serialized if row["event_type"] == "stage" and row["stage"] in STAGE_NAMES}),
                    "measurement_event_count": len(measurement_rows),
                    "kernel_attribution": ("profiler_parent" if any(row["event_type"] == "kernel" for row in measurement_rows) else "trace_timestamp_window"),
                },
            }
        )
        required_markers = {
            "profile/warmup",
            "profile/measure",
            "forward",
            "backward",
            "optimizer",
            "attention/scores",
            "attention/softmax",
            "attention/value",
        }
        seen_markers = set(payload["trace_validation"]["markers_seen"])
        missing_markers = sorted(required_markers - seen_markers)
        if missing_markers:
            raise RuntimeError(f"trace is missing required markers: {', '.join(missing_markers)}")
        if not operators:
            raise RuntimeError("trace contains no measurement-window operator rows")
        if not kernels:
            raise RuntimeError("trace contains no measurement-window CUDA kernel rows")
    except Exception as exc:
        payload.update(
            {
                "status": classify_error(exc),
                "failure_stage": payload.get("failure_stage") or failure_stage,
                "error_type": exc.__class__.__name__,
                "error_summary": safe_error_summary(exc),
                "environment": environment_metadata(torch),
            }
        )
        raise
    finally:
        payload["finished_at"] = utc_now()
        write_json(metadata_path, payload)
    return payload


def main() -> int:
    run_profile(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
