from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from profiling.benchmark import BenchmarkConfig, execute_step
from profiling.config import (
    build_model,
    build_optimizer,
    cuda_memory_metrics,
    model_config,
    public_command,
    public_environment,
    public_path,
    resolve_device,
    sanitize_error,
    seed_everything,
    synchronize,
)
from profiling.nvtx_ranges import instrument_attention, profile_range


def _value(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _operator_rows(profiler: Any) -> list[dict[str, Any]]:
    rows = []
    for event in profiler.key_averages():
        rows.append(
            {
                "kind": "operator",
                "name": event.key,
                "calls": int(getattr(event, "count", 1)),
                "cpu_total_us": round(_value(event, "cpu_time_total"), 3),
                "cpu_self_us": round(_value(event, "self_cpu_time_total"), 3),
                "cuda_total_us": round(
                    _value(event, "device_time_total", "cuda_time_total"), 3
                ),
                "cuda_self_us": round(
                    _value(event, "self_device_time_total", "self_cuda_time_total"), 3
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["cuda_total_us"], row["cpu_total_us"]),
        reverse=True,
    )


def _trace_kernel_rows(trace_path: Path) -> list[dict[str, Any]]:
    try:
        trace = json.loads(trace_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"kind": "kernel", "name": "", "calls": 0, "cuda_total_us": 0.0}
    )
    for event in trace.get("traceEvents", []):
        category = str(event.get("cat", "")).lower()
        name = str(event.get("name", ""))
        if event.get("ph") != "X" or "kernel" not in category:
            continue
        duration = float(event.get("dur", 0.0))
        row = grouped[name]
        row["name"] = name
        row["calls"] += 1
        row["cuda_total_us"] += duration
    return sorted(
        (
            {
                **row,
                "cuda_total_us": round(row["cuda_total_us"], 3),
            }
            for row in grouped.values()
        ),
        key=lambda row: row["cuda_total_us"],
        reverse=True,
    )


def _range_rows(trace_path: Path) -> list[dict[str, Any]]:
    try:
        trace = json.loads(trace_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    ranges: dict[str, dict[str, Any]] = {}
    for event in trace.get("traceEvents", []):
        name = str(event.get("name", ""))
        if not (
            name.startswith("forward")
            or name.startswith("backward")
            or name.startswith("optimizer")
            or name.startswith("attention/")
            or name.startswith("profile/")
        ):
            continue
        if event.get("ph") != "X":
            continue
        row = ranges.setdefault(
            name,
            {
                "kind": "range",
                "name": name,
                "calls": 0,
                "cpu_total_us": 0.0,
                "cuda_total_us": 0.0,
            },
        )
        row["calls"] += 1
        row["cpu_total_us"] += float(event.get("dur", 0.0))
    return sorted(
        (
            {
                **row,
                "cpu_total_us": round(row["cpu_total_us"], 3),
                "cuda_total_us": None,
            }
            for row in ranges.values()
        ),
        key=lambda row: row["cpu_total_us"],
        reverse=True,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "name",
        "calls",
        "cpu_total_us",
        "cpu_self_us",
        "cuda_total_us",
        "cuda_self_us",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_profile(
    config: BenchmarkConfig,
    *,
    trace_path: str | Path,
    summary_path: str | Path,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if config.mode != "train_step":
        raise ValueError("compute profiling must use mode=train_step")
    device = resolve_device(config.device)
    seed_everything(config.seed)
    trace_path = Path(trace_path)
    summary_path = Path(summary_path)
    result: dict[str, Any] = {
        "status": "success",
        "experiment": "compute_profile",
        "config": {
            **config.__dict__,
            "mode": "train_step",
        },
        "model_config": model_config(config.model_size, config.context_length),
        "tool": {
            "name": "torch.profiler",
            "activities": ["CPU", "CUDA"] if device.type == "cuda" else ["CPU"],
            "captures_one_measurement_step": True,
            "trace_file": trace_path.name,
        },
        "command": command,
        "environment": public_environment(device),
        "trace_path": public_path(trace_path),
        "summary_path": public_path(summary_path),
    }

    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(config.model_size, config.context_length, device)
        model.train()
        optimizer = build_optimizer(model)
        cfg = model_config(config.model_size, config.context_length)
        tokens = torch.randint(
            0,
            cfg["vocab_size"],
            (config.batch_size, config.context_length),
            device=device,
        )
        targets = torch.randint_like(tokens, high=cfg["vocab_size"])

        for _ in range(config.warmup):
            execute_step(
                model,
                optimizer,
                tokens,
                targets,
                "train_step",
                config.dtype,
                device,
            )
            synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        trace_path.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with instrument_attention(), torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            with profile_range("profile/warmup"):
                synchronize(device)
            with profile_range("profile/measure"):
                execute_step(
                    model,
                    optimizer,
                    tokens,
                    targets,
                    "train_step",
                    config.dtype,
                    device,
                )
                synchronize(device)
        profiler.export_chrome_trace(str(trace_path))

        operator_rows = _operator_rows(profiler)
        kernel_rows = _trace_kernel_rows(trace_path)
        range_rows = _range_rows(trace_path)
        summary_rows = operator_rows[:30] + kernel_rows[:30] + range_rows
        _write_csv(summary_path.with_suffix(".csv"), summary_rows)
        result["summary"] = {
            "operators": operator_rows[:30],
            "kernels": kernel_rows[:30],
            "ranges": range_rows,
            "csv": public_path(summary_path.with_suffix(".csv")),
        }
        result["memory"] = cuda_memory_metrics(device)
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["exception_type"] = type(exc).__name__
        result["error"] = sanitize_error(exc)
        result["memory"] = cuda_memory_metrics(device)
    except Exception as exc:
        result["status"] = "failed"
        result["exception_type"] = type(exc).__name__
        result["error"] = sanitize_error(exc)
        result["memory"] = cuda_memory_metrics(device)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P torch.profiler run")
    parser.add_argument("--model-size", choices=("small", "medium"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--summary", required=True)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    config = BenchmarkConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode="train_step",
        warmup=args.warmup,
        steps=1,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    cli_arguments = list(arguments) if arguments is not None else sys.argv[1:]
    result = run_profile(
        config,
        trace_path=args.trace,
        summary_path=args.summary,
        command=public_command("profiling.compute_profile", cli_arguments),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"success", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
