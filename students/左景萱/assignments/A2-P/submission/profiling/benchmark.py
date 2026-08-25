#!/usr/bin/env python3
"""Unified A2-P end-to-end benchmark and torch.profiler entry point."""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from profiling.nvtx_ranges import (
    BACKWARD,
    FORWARD,
    OPTIMIZER,
    PROFILE_MEASURE,
    PROFILE_WARMUP,
    annotated_range,
    instrument_attention,
    is_annotation_name,
)


SCHEMA_VERSION = "cs336.a2p.benchmark.v1"
STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
MODES = ("forward", "forward_backward", "train_step")
DTYPES = ("fp32", "bf16")
MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}
TINY_CONFIG = {"d_model": 32, "d_ff": 64, "num_layers": 2, "num_heads": 4}
TRACE_SUMMARY_FIELDS = (
    "scope",
    "op_or_kernel",
    "name",
    "calls",
    "CPU_us",
    "CUDA_us",
    "attribution",
)


@dataclass(frozen=True)
class StepMeasurement:
    phases_ms: dict[str, float]
    total_ms: float
    loss: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the pinned CS336 starter Transformer for A2-P.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--mode",
        action="append",
        choices=MODES,
        help="repeat to select modes; normal benchmark defaults to all three",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--steps",
        type=int,
        help="defaults to 10 normally and exactly 1 with --profile torch",
    )
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--profile", choices=("none", "torch"), default="none")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a tiny CPU matrix; results are explicitly non-authoritative",
    )
    return parser


def _effective_modes(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    selected = list(dict.fromkeys(args.mode or []))
    if args.profile == "torch":
        if selected and selected != ["train_step"]:
            parser.error("--profile torch accepts only --mode train_step")
        return ["train_step"]
    return selected or list(MODES)


def parse_args(
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = _effective_modes(args, parser)
    args.steps = args.steps if args.steps is not None else (1 if args.profile == "torch" else 10)
    if args.batch_size <= 0 or args.context_length <= 0 or args.vocab_size <= 1:
        parser.error("batch/context must be positive and vocab size must exceed one")
    if args.warmup < 0 or args.steps <= 0:
        parser.error("warmup must be non-negative and steps must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.profile == "torch" and (args.warmup < 5 or args.steps != 1):
        parser.error("--profile torch requires at least 5 warmups and exactly 1 measured step")
    return args, modes


def _resolve_device(requested: str, dry_run: bool) -> torch.device:
    if dry_run:
        return torch.device("cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --dry-run for CPU validation")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, dtype: str):
    if dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def execute_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    dtype: str,
    device: torch.device,
    record_function: bool,
) -> StepMeasurement:
    """Run one step with explicit, synchronized phase and total boundaries."""

    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if mode == "train_step" and optimizer is None:
        raise ValueError("train_step requires an optimizer")

    phases_ms: dict[str, float] = {}
    cuda_phase_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    def timed_phase(name: str, action: Any) -> Any:
        if device.type == "cuda":
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
            value = action()
            ended.record()
            cuda_phase_events[name] = (started, ended)
            return value
        started_at = time.perf_counter()
        value = action()
        phases_ms[name] = (time.perf_counter() - started_at) * 1_000.0
        return value

    if mode == "forward_backward":
        model.zero_grad(set_to_none=True)
        _synchronize(device)

    _synchronize(device)
    total_started = time.perf_counter()

    if mode == "train_step":
        assert optimizer is not None
        timed_phase(
            "zero_grad",
            lambda: optimizer.zero_grad(set_to_none=True),
        )

    def forward_action() -> torch.Tensor:
        with annotated_range(
            FORWARD,
            device=device,
            record_function=record_function,
        ):
            if mode == "forward":
                with torch.inference_mode(), _autocast(device, dtype):
                    return model(input_ids)
            with _autocast(device, dtype):
                return model(input_ids)

    logits = timed_phase("forward", forward_action)

    if mode != "forward":

        def loss_action() -> torch.Tensor:
            with _autocast(device, dtype):
                return F.cross_entropy(
                    step_logits.reshape(-1, step_logits.shape[-1]),
                    targets.reshape(-1),
                )

        step_logits = logits
        loss = timed_phase("loss", loss_action)
        assert loss is not None
        backward_loss = loss

        def backward_action() -> None:
            with annotated_range(
                BACKWARD,
                device=device,
                record_function=record_function,
            ):
                backward_loss.backward()

        timed_phase("backward", backward_action)

    if mode == "train_step":
        assert optimizer is not None

        def optimizer_action() -> None:
            with annotated_range(
                OPTIMIZER,
                device=device,
                record_function=record_function,
            ):
                optimizer.step()

        timed_phase("optimizer", optimizer_action)

    _synchronize(device)
    total_ms = (time.perf_counter() - total_started) * 1_000.0
    for name, (started, ended) in cuda_phase_events.items():
        phases_ms[name] = float(started.elapsed_time(ended))
    loss_value = float(loss.detach().float().cpu()) if loss is not None else None
    del logits, loss
    return StepMeasurement(phases_ms=phases_ms, total_ms=total_ms, loss=loss_value)


def _statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty measurement list")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else None
    cv = sample_std / mean if sample_std is not None and mean else None
    return {
        "raw_ms": values,
        "samples": len(values),
        "mean_ms": mean,
        "sample_std_ms": sample_std,
        "cv": cv,
        "cv_percent": None if cv is None else 100.0 * cv,
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


def _memory_peaks(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": None,
            "peak_active_bytes": None,
            "peak_reserved_bytes": None,
        }
    stats = torch.cuda.memory_stats(device)
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_active_bytes": stats.get("active_bytes.all.peak"),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _reset_memory_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _measurement_payload(
    mode: str,
    measurements: list[StepMeasurement],
    memory: dict[str, int | None],
) -> dict[str, Any]:
    phase_names = sorted({name for item in measurements for name in item.phases_ms})
    phase_statistics = {name: _statistics([item.phases_ms[name] for item in measurements]) for name in phase_names}
    total_statistics = _statistics([item.total_ms for item in measurements])
    raw_steps = [
        {
            "step": index,
            "phases_ms": item.phases_ms,
            "total_ms": item.total_ms,
            "loss": item.loss,
        }
        for index, item in enumerate(measurements, start=1)
    ]
    return {
        "mode": mode,
        "status": "ok",
        "raw_steps": raw_steps,
        "phase_statistics": phase_statistics,
        "total_statistics": total_statistics,
        "loss_trend": [item.loss for item in measurements if item.loss is not None],
        "memory": memory,
        "timing_boundary": (
            "zero_grad + forward + loss + backward + optimizer" if mode == "train_step" else "forward + loss + backward" if mode == "forward_backward" else "inference forward only"
        ),
    }


def _make_model_and_data(
    *,
    config: dict[str, int],
    vocab_size: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
    mode: str,
    learning_rate: float,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None, torch.Tensor, torch.Tensor]:
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device)
    model.train(mode != "forward")
    optimizer = AdamW(model.parameters(), lr=learning_rate) if mode == "train_step" else None
    input_ids = torch.randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
    )
    targets = torch.randint(
        vocab_size,
        (batch_size, context_length),
        device=device,
    )
    return model, optimizer, input_ids, targets


def _run_normal_mode(
    *,
    mode: str,
    args: argparse.Namespace,
    config: dict[str, int],
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(args.seed)
    model, optimizer, input_ids, targets = _make_model_and_data(
        config=config,
        vocab_size=vocab_size,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        mode=mode,
        learning_rate=args.learning_rate,
    )
    with instrument_attention(device=device, record_function=False):
        with annotated_range(
            PROFILE_WARMUP,
            device=device,
            record_function=False,
        ):
            for _ in range(args.warmup):
                execute_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    targets=targets,
                    mode=mode,
                    dtype=args.dtype,
                    device=device,
                    record_function=False,
                )
        _reset_memory_peaks(device)
        measurements: list[StepMeasurement] = []
        with annotated_range(
            PROFILE_MEASURE,
            device=device,
            record_function=False,
        ):
            for _ in range(args.steps):
                measurements.append(
                    execute_step(
                        model=model,
                        optimizer=optimizer,
                        input_ids=input_ids,
                        targets=targets,
                        mode=mode,
                        dtype=args.dtype,
                        device=device,
                        record_function=False,
                    )
                )
        memory = _memory_peaks(device)
    del optimizer, model, input_ids, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _synchronize(device)
    return _measurement_payload(mode, measurements, memory)


def _profile_paths(output: Path) -> tuple[Path, Path]:
    stem = output.stem
    return (
        output.with_name(f"{stem}.trace.json"),
        output.with_name(f"{stem}.trace_summary.csv"),
    )


def _profiler_cuda_time_us(item: Any) -> float:
    for field in ("device_time_total", "cuda_time_total"):
        value = getattr(item, field, None)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return 0.0


def _profiler_summary_rows(
    profiler: torch.profiler.profile,
    measurement: StepMeasurement,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in profiler.key_averages():
        name = str(item.key)
        rows.append(
            {
                "scope": "profiler_native",
                "op_or_kernel": "annotation" if is_annotation_name(name) else "operator",
                "name": name,
                "calls": int(item.count),
                "CPU_us": float(item.cpu_time_total),
                "CUDA_us": _profiler_cuda_time_us(item),
                "attribution": "torch.profiler native aggregate",
            }
        )
    wall_values = {**measurement.phases_ms, "total": measurement.total_ms}
    for name, milliseconds in wall_values.items():
        rows.append(
            {
                "scope": "synchronized_phase_wall",
                "op_or_kernel": "phase_wall",
                "name": name,
                "calls": 1,
                "CPU_us": milliseconds * 1_000.0,
                "CUDA_us": None,
                "attribution": ("synchronized wall clock; CUDA_us intentionally blank and is not profiler CUDA attribution"),
            }
        )
    return sorted(rows, key=lambda row: (str(row["scope"]), str(row["name"])))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in TRACE_SUMMARY_FIELDS})
    os.replace(temporary, path)


def _run_torch_profile(
    *,
    args: argparse.Namespace,
    config: dict[str, int],
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(args.seed)
    model, optimizer, input_ids, targets = _make_model_and_data(
        config=config,
        vocab_size=vocab_size,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        mode="train_step",
        learning_rate=args.learning_rate,
    )
    assert optimizer is not None
    trace_path, summary_path = _profile_paths(output)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with instrument_attention(device=device, record_function=True):
        for _ in range(args.warmup):
            execute_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                mode="train_step",
                dtype=args.dtype,
                device=device,
                record_function=False,
            )
        with torch.profiler.profile(activities=activities) as profiler:
            with annotated_range(
                PROFILE_WARMUP,
                device=device,
                record_function=True,
            ):
                # The actual warm-up steps intentionally ran before profiler
                # collection.  Keep this boundary marker in the trace without
                # contaminating the exactly-one-step operator aggregates.
                _synchronize(device)
            _reset_memory_peaks(device)
            with annotated_range(
                PROFILE_MEASURE,
                device=device,
                record_function=True,
            ):
                measurement = execute_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    targets=targets,
                    mode="train_step",
                    dtype=args.dtype,
                    device=device,
                    record_function=True,
                )
            memory = _memory_peaks(device)
        profiler.export_chrome_trace(str(trace_path))
        summary_rows = _profiler_summary_rows(profiler, measurement)
        _atomic_csv(summary_path, summary_rows)

    result = _measurement_payload("train_step", [measurement], memory)
    profile = {
        "tool": "torch.profiler",
        "activities": [activity.name for activity in activities],
        "warmup_steps_before_measurement": args.warmup,
        "warmup_steps_in_trace": 0,
        "measured_steps": 1,
        "trace_file": trace_path.name,
        "summary_file": summary_path.name,
        "summary_columns": list(TRACE_SUMMARY_FIELDS),
        "native_cuda_attribution": device.type == "cuda",
        "wall_row_guard": ("synchronized_phase_wall rows leave CUDA_us blank; they supplement cross-thread phase boundaries and are not profiler CUDA attribution"),
    }
    del optimizer, model, input_ids, targets
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _synchronize(device)
    return result, profile


def _cuda_driver_version() -> str | None:
    for owner, name in (
        (torch.cuda, "driver_version"),
        (torch._C, "_cuda_getDriverVersion"),
    ):
        function = getattr(owner, name, None)
        if callable(function):
            try:
                value = function()
            except (RuntimeError, OSError):
                continue
            if type(value) is int:
                return str(value)
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    first_line = completed.stdout.splitlines()[0].strip() if completed.stdout else ""
    return first_line if re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", first_line) else None


def _safe_metadata(
    *,
    args: argparse.Namespace,
    modes: list[str],
    device: torch.device,
    output: Path,
    dry_run: bool,
) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "model": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    logical_arguments = [
        "--model-size",
        args.model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--learning-rate",
        str(args.learning_rate),
        "--vocab-size",
        str(args.vocab_size),
        "--profile",
        args.profile,
        "--output",
        f"results/{output.name}",
    ]
    for mode in modes:
        logical_arguments.extend(("--mode", mode))
    if dry_run:
        logical_arguments.append("--dry-run")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "starter_commit": STARTER_COMMIT,
        "command": ["python", "profiling/benchmark.py", *logical_arguments],
        "result_file": output.name,
        "hardware": {
            "device_type": device.type,
            "gpu": gpu,
            "cuda_driver_version": _cuda_driver_version() if gpu is not None else None,
        },
        "software": {
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "profiler": "torch.profiler" if args.profile == "torch" else None,
        },
        "privacy": {
            "public_allowlist": True,
        },
    }


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args, modes = parse_args(argv)
    output = args.output.expanduser().resolve()
    device = _resolve_device(args.device, args.dry_run)
    config = dict(TINY_CONFIG if args.dry_run else MODEL_CONFIGS[args.model_size])
    batch_size = 1 if args.dry_run else args.batch_size
    context_length = 8 if args.dry_run else args.context_length
    vocab_size = 64 if args.dry_run else args.vocab_size
    torch.set_float32_matmul_precision("highest")

    results: list[dict[str, Any]] = []
    profile_payload: dict[str, Any] | None = None
    if args.profile == "torch":
        result, profile_payload = _run_torch_profile(
            args=args,
            config=config,
            batch_size=batch_size,
            context_length=context_length,
            vocab_size=vocab_size,
            device=device,
            output=output,
        )
        results.append(result)
    else:
        for mode in modes:
            results.append(
                _run_normal_mode(
                    mode=mode,
                    args=args,
                    config=config,
                    batch_size=batch_size,
                    context_length=context_length,
                    vocab_size=vocab_size,
                    device=device,
                )
            )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "authoritative": device.type == "cuda" and not args.dry_run,
        "non_authoritative_reason": ("CPU tiny dry-run validates control flow only" if args.dry_run else "CPU timings are diagnostic only" if device.type != "cuda" else None),
        "metadata": _safe_metadata(
            args=args,
            modes=modes,
            device=device,
            output=output,
            dry_run=args.dry_run,
        ),
        "configuration": {
            "requested_model_size": args.model_size,
            "effective_model_size": "tiny" if args.dry_run else args.model_size,
            **config,
            "requested_batch_size": args.batch_size,
            "batch_size": batch_size,
            "requested_context_length": args.context_length,
            "context_length": context_length,
            "vocab_size": vocab_size,
            "modes": modes,
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "dtype": args.dtype,
            "autocast_dtype": "torch.bfloat16" if args.dtype == "bf16" else None,
            "parameter_dtype": "torch.float32",
            "seed": args.seed,
            "device": device.type,
            "learning_rate": args.learning_rate,
        },
        "measurement_contract": {
            "clock": "time.perf_counter",
            "cuda_synchronize_at_step_boundaries": device.type == "cuda",
            "cuda_synchronize_after_each_phase": False,
            "phase_timer": "CUDA events" if device.type == "cuda" else "time.perf_counter",
            "initialization_and_data_generation_timed": False,
            "forward_uses_inference_mode": True,
            "forward_backward_clears_gradients_each_step": True,
            "train_step_total_includes": [
                "zero_grad",
                "forward",
                "loss",
                "backward",
                "optimizer",
            ],
            "standard_deviation": "sample standard deviation (n-1)",
            "cv": "sample_std_ms / mean_ms",
            "memory_peaks": {
                "allocated": "torch.cuda.max_memory_allocated",
                "active": "torch.cuda.memory_stats()['active_bytes.all.peak']",
                "reserved": "torch.cuda.max_memory_reserved",
            },
        },
        "results": results,
        "profile": profile_payload,
    }
    _atomic_json(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    payload = run(argv)
    print(f"status={payload['status']} authoritative={payload['authoritative']}")
    print(f"JSON: {payload['metadata']['result_file']}")
    if isinstance(payload.get("profile"), dict):
        print(f"Trace: {payload['profile']['trace_file']}")
        print(f"Summary: {payload['profile']['summary_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
