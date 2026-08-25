#!/usr/bin/env python3
"""Capture CUDA allocator snapshots for the A2-P memory-profiling matrix.

Raw ``.pickle`` snapshots and optional Chrome traces are deliberately written to a
local artifact directory, not to ``results/memory``.  Only the compact CSV and
JSON metadata in ``results/memory`` are intended to be copied into a public
submission after review.

Example:

    uv run python profiling/memory_snapshot.py \
        --model-size xl --contexts 128 2048 --modes forward train_step

The script uses the shared helpers in ``profiling/benchmark.py`` so that model
construction and workload boundaries match the timing and compute-profiling
experiments.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import datetime as datetime_module
import gc
import importlib
import json
import os
import platform
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


MEBIBYTE = 1024**2
DEFAULT_CONTEXTS = (128, 2048)
DEFAULT_MODES = ("forward", "train_step")
CSV_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "status",
    "requested_model_size",
    "requested_context_length",
    "requested_batch_size",
    "requested_mode",
    "model_size",
    "context_length",
    "batch_size",
    "mode",
    "dtype",
    "seed",
    "warmup_steps",
    "fallback_level",
    "fallback_reason",
    "measurement_elapsed_ms",
    "memory_history_status",
    "torch_profiler_memory_enabled",
    "torch_profiler_memory_status",
    "snapshot_saved",
    "snapshot_file",
    "profiler_trace_file",
    "active_bytes_after_step",
    "active_mib_after_step",
    "allocated_bytes_after_step",
    "allocated_mib_after_step",
    "reserved_bytes_after_step",
    "reserved_mib_after_step",
    "active_bytes_peak",
    "active_mib_peak",
    "allocated_bytes_peak",
    "allocated_mib_peak",
    "reserved_bytes_peak",
    "reserved_mib_peak",
    "peak_allocated_bytes",
    "peak_allocated_mib",
    "peak_reserved_bytes",
    "peak_reserved_mib",
    "failure_stage",
    "exception_type",
    "error_message",
    "warning_stages",
    "warning_message",
)


@dataclasses.dataclass(frozen=True)
class AttemptSpec:
    """One requested or explicitly labelled fallback measurement."""

    requested_model_size: str
    requested_context_length: int
    requested_batch_size: int
    requested_mode: str
    model_size: str
    context_length: int
    batch_size: int
    mode: str
    fallback_level: int = 0
    fallback_reason: str | None = None


@dataclasses.dataclass
class BenchmarkApi:
    """The small stable surface imported from ``profiling.benchmark``."""

    build_model: Any
    make_random_batch: Any
    create_optimizer: Any
    execute_workload: Any
    synchronize: Any


@dataclasses.dataclass
class AttemptOutcome:
    row: dict[str, Any]
    profiler_top_ops: list[dict[str, Any]]
    local_snapshot: Path | None
    local_profiler_trace: Path | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return parsed


def default_snapshot_dir() -> Path:
    """Choose a local-only directory that is outside the Git worktree by default."""

    return Path(tempfile.gettempdir()) / "cs336-a2p-memory-snapshots"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="采集 Transformer CUDA memory history snapshot，并写出轻量峰值汇总。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-size", default="xl", help="主实验模型规模；A2-P 默认 xl")
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_CONTEXTS),
        help="待测 context length；默认覆盖 A2-P 的 128 和 2048",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
        help="forward 为 inference-only；train_step 包含 forward/backward/optimizer",
    )
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--vocab-size", type=positive_int, default=10_000)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=nonnegative_int, default=1, help="开启 memory history 前的完整 warm-up step 数")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda", help="CUDA device，例如 cuda 或 cuda:0")
    parser.add_argument("--history-max-entries", type=positive_int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "memory")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=default_snapshot_dir(),
        help="本地原始 snapshot/trace 目录；不得提交或复制到公开 results/",
    )
    parser.add_argument(
        "--torch-profiler-memory",
        action="store_true",
        help="额外运行 torch.profiler(profile_memory=True)，原始 Chrome trace 仍只保留在本地",
    )
    parser.add_argument(
        "--oom-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="XL/context=2048 OOM 时按题目要求尝试 batch=1、XL/1024、Large/2048，并保留每次失败记录",
    )
    parser.add_argument("--fallback-context", type=positive_int, default=1024)
    parser.add_argument("--fallback-model-size", default="large")
    parser.add_argument("--append", action="store_true", help="向已有 peaks.csv 追加；默认覆盖本次汇总")
    parser.add_argument("--fail-fast", action="store_true", help="第一项非 OOM 错误后立即停止")
    return parser.parse_args(argv)


def utc_now() -> datetime_module.datetime:
    return datetime_module.datetime.now(datetime_module.timezone.utc)


def utc_timestamp() -> str:
    return utc_now().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def run_id_for(timestamp: datetime_module.datetime, ordinal: int) -> str:
    """Use a timestamp rather than a UUID: public metadata must not contain UUIDs."""

    compact = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return f"memory-{compact}-{ordinal:02d}"


def redact_message(value: object, limit: int = 500) -> str:
    """Keep failure metadata useful without leaking local paths, IPs, or UUIDs."""

    text = " ".join(str(value).split())
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", text)
    text = re.sub(
        r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b",
        "<id>",
        text,
    )
    # CUDA/Python errors can include a source or local filesystem path.  The path
    # is not needed to reproduce the configuration and is unsafe in public JSON.
    text = re.sub(r"(?<![\w<])/(?:[^\s:'\"()\[\],]+/)*[^\s:'\"()\[\],]+", "<path>", text)
    return text[:limit]


def as_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / MEBIBYTE, 3)


def optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_cuda_oom(error: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "cudnn" in message or "allocator" in message)


def load_benchmark_api() -> BenchmarkApi:
    """Load the shared benchmark helpers for both direct-script and module execution."""

    import_errors: list[str] = []
    module: ModuleType | None = None
    for module_name in ("benchmark", "profiling.benchmark"):
        try:
            module = importlib.import_module(module_name)
            break
        except ModuleNotFoundError as error:
            # Only suppress a missing module at the import boundary.  A missing
            # dependency *inside* benchmark.py should be shown to the caller.
            if error.name == module_name or error.name in {"profiling", "benchmark"}:
                import_errors.append(f"{module_name}: {error}")
                continue
            raise
    if module is None:
        joined = "; ".join(import_errors)
        raise RuntimeError(f"无法导入 profiling/benchmark.py 的共享接口：{joined}")

    required = ("build_model", "make_random_batch", "create_optimizer", "execute_workload", "synchronize")
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"profiling/benchmark.py 缺少可调用接口：{missing_text}")
    return BenchmarkApi(**{name: getattr(module, name) for name in required})


def cuda_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "torch_version": torch.__version__,
        "compiled_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if environment["cuda_available"] and device.type == "cuda":
        try:
            properties = torch.cuda.get_device_properties(device)
            environment.update(
                {
                    "gpu_name": torch.cuda.get_device_name(device),
                    "gpu_total_memory_bytes": int(properties.total_memory),
                    "gpu_total_memory_mib": as_mib(int(properties.total_memory)),
                }
            )
        except Exception as error:  # pragma: no cover - depends on CUDA runtime state
            environment["gpu_query_error"] = redact_message(error)
    return environment


def reset_cuda_allocator(device: torch.device) -> None:
    """Release dead tensors/cached blocks before beginning an independent attempt."""

    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.synchronize(device)
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.reset_peak_memory_stats(device)


def collect_memory_counters(device: torch.device) -> dict[str, int | None]:
    """Read distinct allocator counters without conflating active/allocated/reserved."""

    try:
        stats = torch.cuda.memory_stats(device)
        allocated_current = optional_int(torch.cuda.memory_allocated(device))
        reserved_current = optional_int(torch.cuda.memory_reserved(device))
        peak_allocated = optional_int(torch.cuda.max_memory_allocated(device))
        peak_reserved = optional_int(torch.cuda.max_memory_reserved(device))
    except Exception:
        return {
            "active_current": None,
            "allocated_current": None,
            "reserved_current": None,
            "active_peak": None,
            "allocated_peak": None,
            "reserved_peak": None,
            "peak_allocated": None,
            "peak_reserved": None,
        }

    return {
        "active_current": optional_int(stats.get("active_bytes.all.current")),
        "allocated_current": allocated_current,
        "reserved_current": reserved_current,
        "active_peak": optional_int(stats.get("active_bytes.all.peak")),
        "allocated_peak": optional_int(stats.get("allocated_bytes.all.peak")),
        "reserved_peak": optional_int(stats.get("reserved_bytes.all.peak")),
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved,
    }


def add_memory_fields(row: dict[str, Any], counters: dict[str, int | None]) -> None:
    mappings = {
        "active_bytes_after_step": counters["active_current"],
        "allocated_bytes_after_step": counters["allocated_current"],
        "reserved_bytes_after_step": counters["reserved_current"],
        "active_bytes_peak": counters["active_peak"],
        "allocated_bytes_peak": counters["allocated_peak"],
        "reserved_bytes_peak": counters["reserved_peak"],
        "peak_allocated_bytes": counters["peak_allocated"],
        "peak_reserved_bytes": counters["peak_reserved"],
    }
    for column, value in mappings.items():
        row[column] = value
        row[column.replace("_bytes", "_mib")] = as_mib(value)


def enable_memory_history(device: torch.device, max_entries: int) -> str | None:
    """Enable PyTorch history across several supported private-API signatures."""

    recorder = torch.cuda.memory._record_memory_history
    try:
        recorder(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=max_entries,
            device=device,
            clear_history=True,
        )
        return None
    except TypeError:
        # Older PyTorch releases expose the same profiler with fewer keyword
        # arguments.  It remains preferable to fail open with labelled metadata
        # than to omit the measurement entirely.
        try:
            recorder(enabled="all", max_entries=max_entries, device=device)
            return None
        except TypeError:
            try:
                recorder(max_entries=max_entries)
                return None
            except Exception as error:  # pragma: no cover - version-specific
                return redact_message(error)
        except Exception as error:  # pragma: no cover - runtime-specific
            return redact_message(error)
    except Exception as error:  # pragma: no cover - runtime-specific
        return redact_message(error)


def disable_memory_history(device: torch.device) -> str | None:
    recorder = torch.cuda.memory._record_memory_history
    try:
        recorder(enabled=None, device=device)
        return None
    except TypeError:
        try:
            recorder(enabled=None)
            return None
        except Exception as error:  # pragma: no cover - version-specific
            return redact_message(error)
    except Exception as error:  # pragma: no cover - runtime-specific
        return redact_message(error)


def dump_snapshot(snapshot_path: Path) -> str | None:
    try:
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        return None
    except Exception as error:  # pragma: no cover - depends on CUDA/CUPTI build
        return redact_message(error)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    with contextlib.suppress(Exception):
        torch.cuda.manual_seed_all(seed)


def snapshot_filename(run_id: str, spec: AttemptSpec) -> str:
    return (
        f"{run_id}__requested-{spec.requested_model_size}-ctx{spec.requested_context_length}-bs{spec.requested_batch_size}"
        f"__actual-{spec.model_size}-ctx{spec.context_length}-bs{spec.batch_size}-{spec.mode}.pickle"
    )


def profiler_trace_filename(snapshot_file: str) -> str:
    return f"{Path(snapshot_file).stem}.memory-profiler.json"


def profile_memory_top_ops(profiler: Any, limit: int = 5) -> list[dict[str, Any]]:
    """Return a compact, public-safe allocation summary rather than a raw trace."""

    operations: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        self_cuda_bytes = optional_int(getattr(event, "self_cuda_memory_usage", None)) or 0
        total_cuda_bytes = optional_int(getattr(event, "cuda_memory_usage", None)) or 0
        self_cpu_bytes = optional_int(getattr(event, "self_cpu_memory_usage", None)) or 0
        total_cpu_bytes = optional_int(getattr(event, "cpu_memory_usage", None)) or 0
        if max(abs(self_cuda_bytes), abs(total_cuda_bytes), abs(self_cpu_bytes), abs(total_cpu_bytes)) == 0:
            continue
        operations.append(
            {
                "op_name": str(getattr(event, "key", "<unknown>")),
                "calls": optional_int(getattr(event, "count", None)) or 0,
                "self_cuda_memory_bytes": self_cuda_bytes,
                "cuda_memory_bytes": total_cuda_bytes,
                "self_cpu_memory_bytes": self_cpu_bytes,
                "cpu_memory_bytes": total_cpu_bytes,
            }
        )
    return sorted(
        operations,
        key=lambda item: (abs(item["self_cuda_memory_bytes"]), abs(item["cuda_memory_bytes"])),
        reverse=True,
    )[:limit]


def execute_measurement(
    api: BenchmarkApi,
    spec: AttemptSpec,
    model: Any,
    optimizer: Any,
    input_ids: Any,
    targets: Any,
    device: torch.device,
    dtype: torch.dtype,
    use_profiler: bool,
    profiler_trace_path: Path,
) -> tuple[float, str, str | None, list[dict[str, Any]], bool]:
    """Run one measured workload, optionally preserving a local profiler trace.

    A CUPTI/profiler setup failure does not discard the measurement: the workload
    is rerun without ``torch.profiler`` only when it is known not to have run yet.
    OOM exceptions always propagate to the outer attempt handler.
    """

    def run_workload() -> None:
        with torch.profiler.record_function(f"memory/{spec.mode}"):
            api.execute_workload(spec.mode, model, optimizer, input_ids, targets, device, dtype)

    started_at = time.perf_counter()
    if not use_profiler:
        run_workload()
        api.synchronize(device)
        return (time.perf_counter() - started_at) * 1_000.0, "disabled", None, [], False

    profiler: Any | None = None
    workload_finished = False
    profiler_error: str | None = None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            record_shapes=False,
            with_stack=False,
        ) as profiler:
            run_workload()
            workload_finished = True
        api.synchronize(device)
    except BaseException as error:
        if is_cuda_oom(error):
            raise
        profiler_error = redact_message(error)
        if not workload_finished:
            run_workload()
            workload_finished = True
            api.synchronize(device)
        return (time.perf_counter() - started_at) * 1_000.0, "failed", profiler_error, [], False

    profiler_status = "captured"
    top_ops: list[dict[str, Any]] = []
    trace_saved = False
    try:
        top_ops = profile_memory_top_ops(profiler)
    except Exception as error:  # pragma: no cover - profiler-version specific
        profiler_status = "captured_with_summary_error"
        profiler_error = redact_message(error)
    try:
        profiler.export_chrome_trace(str(profiler_trace_path))
        trace_saved = True
    except Exception as error:  # pragma: no cover - CUPTI/runtime-specific
        profiler_status = "captured_with_trace_error" if profiler_status == "captured" else profiler_status
        profiler_error = redact_message(error)
    return (time.perf_counter() - started_at) * 1_000.0, profiler_status, profiler_error, top_ops, trace_saved


def base_row(run_id: str, timestamp: str, spec: AttemptSpec, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in CSV_COLUMNS}
    row.update(
        {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "status": "pending",
            "requested_model_size": spec.requested_model_size,
            "requested_context_length": spec.requested_context_length,
            "requested_batch_size": spec.requested_batch_size,
            "requested_mode": spec.requested_mode,
            "model_size": spec.model_size,
            "context_length": spec.context_length,
            "batch_size": spec.batch_size,
            "mode": spec.mode,
            "dtype": args.dtype,
            "seed": seed,
            "warmup_steps": args.warmup,
            "fallback_level": spec.fallback_level,
            "fallback_reason": spec.fallback_reason or "",
            "torch_profiler_memory_enabled": args.torch_profiler_memory,
            "snapshot_saved": False,
        }
    )
    return row


def run_attempt(
    api: BenchmarkApi,
    spec: AttemptSpec,
    args: argparse.Namespace,
    device: torch.device,
    run_time: datetime_module.datetime,
    ordinal: int,
) -> AttemptOutcome:
    """Measure one configuration and return a row even when it fails or OOMs."""

    run_id = run_id_for(run_time, ordinal)
    row = base_row(run_id, run_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"), spec, args, args.seed + ordinal)
    snapshot_path = args.snapshot_dir / snapshot_filename(run_id, spec)
    trace_path = args.snapshot_dir / profiler_trace_filename(snapshot_path.name)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    history_enabled = False
    model: Any | None = None
    optimizer: Any | None = None
    input_ids: Any | None = None
    targets: Any | None = None
    profiler_top_ops: list[dict[str, Any]] = []
    local_snapshot: Path | None = None
    local_trace: Path | None = None
    warnings: list[tuple[str, str]] = []
    stage = "setup"

    reset_cuda_allocator(device)
    try:
        set_seed(args.seed + ordinal)
        stage = "initialization"
        model = api.build_model(spec.model_size, spec.context_length, args.vocab_size, device)
        optimizer = api.create_optimizer(model)
        input_ids, targets = api.make_random_batch(spec.batch_size, spec.context_length, args.vocab_size, device)

        stage = "warmup"
        for _ in range(args.warmup):
            api.execute_workload(spec.mode, model, optimizer, input_ids, targets, device, dtype)
            api.synchronize(device)

        # Peak counters must start after warm-up, while the memory history must
        # begin only after warm-up.  This is the key measurement boundary.
        stage = "prepare_measurement"
        api.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        stage = "enable_memory_history"
        history_error = enable_memory_history(device, args.history_max_entries)
        if history_error is None:
            history_enabled = True
            row["memory_history_status"] = "enabled_after_warmup"
        else:
            row["memory_history_status"] = "unavailable"
            warnings.append((stage, history_error))

        stage = "measurement"
        elapsed_ms, profiler_status, profiler_error, profiler_top_ops, trace_saved = execute_measurement(
            api=api,
            spec=spec,
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            device=device,
            dtype=dtype,
            use_profiler=args.torch_profiler_memory,
            profiler_trace_path=trace_path,
        )
        row["measurement_elapsed_ms"] = round(elapsed_ms, 3)
        row["torch_profiler_memory_status"] = profiler_status
        if profiler_error is not None:
            warnings.append(("torch_profiler_memory", profiler_error))
        if trace_saved:
            row["profiler_trace_file"] = trace_path.name
            local_trace = trace_path

        api.synchronize(device)
        add_memory_fields(row, collect_memory_counters(device))

        if history_enabled:
            stage = "snapshot"
            snapshot_error = dump_snapshot(snapshot_path)
            if snapshot_error is None:
                row["snapshot_saved"] = True
                row["snapshot_file"] = snapshot_path.name
                local_snapshot = snapshot_path
            else:
                warnings.append((stage, snapshot_error))

        row["status"] = "completed" if not warnings else "completed_with_warnings"
    except BaseException as error:
        error_is_oom = is_cuda_oom(error)
        row["status"] = "oom" if error_is_oom else "error"
        row["failure_stage"] = stage
        row["exception_type"] = type(error).__name__
        row["error_message"] = redact_message(error)
        row["torch_profiler_memory_status"] = row["torch_profiler_memory_status"] or (
            "not_completed" if args.torch_profiler_memory else "disabled"
        )

        # A post-OOM snapshot can be useful to inspect the failed allocation.  It
        # is best-effort and never masks the original OOM/failure status.
        add_memory_fields(row, collect_memory_counters(device))
        if history_enabled:
            snapshot_error = dump_snapshot(snapshot_path)
            if snapshot_error is None:
                row["snapshot_saved"] = True
                row["snapshot_file"] = snapshot_path.name
                local_snapshot = snapshot_path
            else:
                warnings.append(("snapshot_after_failure", snapshot_error))
    finally:
        if history_enabled:
            disable_error = disable_memory_history(device)
            if disable_error is not None:
                warnings.append(("disable_memory_history", disable_error))
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        if input_ids is not None:
            del input_ids
        if targets is not None:
            del targets
        gc.collect()
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    if row["memory_history_status"] is None:
        row["memory_history_status"] = "not_started"
    if row["torch_profiler_memory_status"] is None:
        row["torch_profiler_memory_status"] = "disabled"
    if warnings:
        row["warning_stages"] = ";".join(stage_name for stage_name, _ in warnings)
        row["warning_message"] = " | ".join(message for _, message in warnings)
        if row["status"] == "completed":
            row["status"] = "completed_with_warnings"
    return AttemptOutcome(row, profiler_top_ops, local_snapshot, local_trace)


def requested_specs(args: argparse.Namespace) -> list[AttemptSpec]:
    return [
        AttemptSpec(
            requested_model_size=args.model_size.lower(),
            requested_context_length=context_length,
            requested_batch_size=args.batch_size,
            requested_mode=mode,
            model_size=args.model_size.lower(),
            context_length=context_length,
            batch_size=args.batch_size,
            mode=mode,
        )
        for context_length in args.contexts
        for mode in args.modes
    ]


def oom_fallback_specs(spec: AttemptSpec, args: argparse.Namespace) -> list[AttemptSpec]:
    """Implement the assignment's labelled XL/2048 fallback sequence.

    No fallback is relabelled as the requested XL/2048 measurement: every CSV
    row retains requested and actual configuration columns plus its reason.
    """

    if not args.oom_fallback or spec.requested_model_size != "xl" or spec.requested_context_length != 2048:
        return []

    candidates: list[tuple[str, int, int, str]] = []
    if spec.requested_batch_size != 1:
        candidates.append(("xl", 2048, 1, "XL/context=2048 原 batch 失败；按题目要求重试 batch=1"))
    candidates.extend(
        [
            ("xl", args.fallback_context, 1, "XL/context=2048、batch=1 OOM；改测 XL/fallback context"),
            (args.fallback_model_size.lower(), 2048, 1, "XL/context=2048、batch=1 OOM；改测 fallback model/context=2048"),
        ]
    )

    fallbacks: list[AttemptSpec] = []
    seen: set[tuple[str, int, int]] = {(spec.model_size, spec.context_length, spec.batch_size)}
    for level, (model_size, context_length, batch_size, reason) in enumerate(candidates, start=1):
        key = (model_size, context_length, batch_size)
        if key in seen:
            continue
        seen.add(key)
        fallbacks.append(
            AttemptSpec(
                requested_model_size=spec.requested_model_size,
                requested_context_length=spec.requested_context_length,
                requested_batch_size=spec.requested_batch_size,
                requested_mode=spec.requested_mode,
                model_size=model_size,
                context_length=context_length,
                batch_size=batch_size,
                mode=spec.mode,
                fallback_level=level,
                fallback_reason=reason,
            )
        )
    return fallbacks


def write_csv(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = rows
    if append and path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise RuntimeError(f"已有 {path.name} 的列与当前脚本不兼容；请移除该文件或不使用 --append")
            all_rows = list(reader) + rows
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_command(args: argparse.Namespace) -> str:
    """Store a reproducible command without local artifact paths or user details."""

    command = [
        "python",
        "profiling/memory_snapshot.py",
        "--model-size",
        args.model_size.lower(),
        "--contexts",
        *(str(value) for value in args.contexts),
        "--modes",
        *args.modes,
        "--batch-size",
        str(args.batch_size),
        "--vocab-size",
        str(args.vocab_size),
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--history-max-entries",
        str(args.history_max_entries),
    ]
    if args.device != "cuda":
        command.extend(("--device", args.device))
    if args.torch_profiler_memory:
        command.append("--torch-profiler-memory")
    if not args.oom_fallback:
        command.append("--no-oom-fallback")
    return shlex.join(command)


def metric_definitions() -> dict[str, str]:
    return {
        "active": "PyTorch allocator 的 active_bytes.all；表示活跃 allocator block，不等同于 tensor allocated。",
        "allocated": "torch.cuda.memory_allocated；当前由活跃 tensor 占用的字节数。",
        "reserved": "torch.cuda.memory_reserved；CUDA caching allocator 向驱动保留的字节数，可能包含可复用空闲块。",
        "peak": "在 warm-up 完成、empty_cache 与 reset_peak_memory_stats 后，到本次 measurement 结束期间的最大值。",
        "units": "CSV 同时提供精确 bytes 与 MiB；1 MiB = 1024^2 bytes。",
    }


def preflight_rows(args: argparse.Namespace, status: str, message: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = utc_now()
    for ordinal, spec in enumerate(requested_specs(args), start=1):
        row = base_row(run_id_for(now, ordinal), utc_timestamp(), spec, args, args.seed + ordinal)
        row.update(
            {
                "status": status,
                "failure_stage": "preflight",
                "exception_type": "CudaUnavailable" if status == "cuda_unavailable" else "InvalidDevice",
                "error_message": message,
                "memory_history_status": "not_started",
                "torch_profiler_memory_status": "not_started" if args.torch_profiler_memory else "disabled",
            }
        )
        rows.append(row)
    return rows


def make_metadata(
    args: argparse.Namespace,
    status: str,
    environment: dict[str, Any],
    outcomes: list[AttemptOutcome],
    extra_error: str | None = None,
) -> dict[str, Any]:
    rows = [outcome.row for outcome in outcomes]
    return {
        "schema_version": 1,
        "kind": "a2p_memory_profile",
        "generated_at_utc": utc_timestamp(),
        "status": status,
        "reproducible_command": canonical_command(args),
        "environment": environment,
        "settings": {
            "model_size": args.model_size.lower(),
            "contexts": args.contexts,
            "modes": args.modes,
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "dtype": args.dtype,
            "warmup_steps": args.warmup,
            "seed": args.seed,
            "memory_history_max_entries": args.history_max_entries,
            "torch_profiler_memory": args.torch_profiler_memory,
        },
        "measurement_boundary": "warm-up 完成后才启用 memory history；随后 reset peak 后只执行一个 measurement workload。",
        "metric_definitions": metric_definitions(),
        "oom_fallback_policy": {
            "enabled": args.oom_fallback,
            "scope": "仅 requested XL/context=2048",
            "sequence": [
                "XL/context=2048/batch=1（原 batch 大于 1 时）",
                f"XL/context={args.fallback_context}/batch=1",
                f"{args.fallback_model_size.lower()}/context=2048/batch=1",
            ],
            "labelling": "每个 fallback 行同时保留 requested_* 与实际配置字段，不会伪装为 XL/context=2048。",
        },
        "raw_artifact_policy": {
            "snapshot_and_trace_location": "仅本地 artifact 目录；为避免公开路径泄露，metadata 不写目录路径。",
            "do_not_commit": ["*.pickle", "*.json Chrome trace"],
            "snapshot_files": [outcome.row["snapshot_file"] for outcome in outcomes if outcome.row.get("snapshot_file")],
            "profiler_trace_files": [outcome.row["profiler_trace_file"] for outcome in outcomes if outcome.row.get("profiler_trace_file")],
        },
        "counts": {
            "attempts": len(rows),
            "completed": sum(str(row["status"]).startswith("completed") for row in rows),
            "oom": sum(row["status"] == "oom" for row in rows),
            "errors": sum(row["status"] == "error" for row in rows),
        },
        "runs": [
            {
                "metrics": outcome.row,
                "torch_profiler_top_memory_ops": outcome.profiler_top_ops,
            }
            for outcome in outcomes
        ],
        **({"error": extra_error} if extra_error else {}),
    }


def print_attempt_summary(outcome: AttemptOutcome) -> None:
    row = outcome.row
    requested = f"{row['requested_model_size']}/ctx{row['requested_context_length']}/bs{row['requested_batch_size']}/{row['requested_mode']}"
    actual = f"{row['model_size']}/ctx{row['context_length']}/bs{row['batch_size']}/{row['mode']}"
    peak = row.get("peak_allocated_mib")
    peak_text = "n/a" if peak is None else f"{peak:.3f} MiB"
    print(f"[{row['status']}] requested={requested}; actual={actual}; peak_allocated={peak_text}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.model_size = args.model_size.lower()
    args.fallback_model_size = args.fallback_model_size.lower()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    peaks_path = args.output_dir / "peaks.csv"
    metadata_path = args.output_dir / "run_metadata.json"

    try:
        device = torch.device(args.device)
    except (TypeError, RuntimeError) as error:
        message = f"无效 device：{redact_message(error)}"
        rows = preflight_rows(args, "invalid_device", message)
        outcomes = [AttemptOutcome(row, [], None, None) for row in rows]
        write_csv(peaks_path, rows, args.append)
        write_json(metadata_path, make_metadata(args, "invalid_device", {}, outcomes, message))
        print(message, file=sys.stderr)
        return 2

    if device.type != "cuda":
        message = "memory history snapshot 仅支持 CUDA；请使用 --device cuda 或 cuda:N。"
        rows = preflight_rows(args, "invalid_device", message)
        outcomes = [AttemptOutcome(row, [], None, None) for row in rows]
        write_csv(peaks_path, rows, args.append)
        write_json(metadata_path, make_metadata(args, "invalid_device", cuda_environment(device), outcomes, message))
        print(message, file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        message = (
            "当前 PyTorch 进程未检测到可用 CUDA。无法采集 CUDA allocator history 或 snapshot；"
            f"torch={torch.__version__}，compiled_cuda={torch.version.cuda}。请在可用 NVIDIA CUDA 环境重试。"
        )
        rows = preflight_rows(args, "cuda_unavailable", message)
        outcomes = [AttemptOutcome(row, [], None, None) for row in rows]
        write_csv(peaks_path, rows, args.append)
        write_json(metadata_path, make_metadata(args, "cuda_unavailable", cuda_environment(device), outcomes, message))
        print(message, file=sys.stderr)
        return 2

    try:
        
        if device.index is not None:
            torch.cuda.set_device(device)
        environment = cuda_environment(device)
        api = load_benchmark_api()
    except BaseException as error:
        message = redact_message(error)
        rows = preflight_rows(args, "setup_error", message)
        outcomes = [AttemptOutcome(row, [], None, None) for row in rows]
        write_csv(peaks_path, rows, args.append)
        write_json(metadata_path, make_metadata(args, "setup_error", cuda_environment(device), outcomes, message))
        print(f"初始化 memory profiling 失败：{message}", file=sys.stderr)
        return 2

    # The raw directory is separate from public output and intentionally omitted
    # from run_metadata.json.  It may be supplied explicitly for a shared local
    # scratch disk, but should never be copied into a submission.
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[AttemptOutcome] = []
    ordinal = 0
    for requested in requested_specs(args):
        attempts = [requested, *oom_fallback_specs(requested, args)]
        for index, attempt in enumerate(attempts):
            ordinal += 1
            outcome = run_attempt(api, attempt, args, device, utc_now(), ordinal)
            outcomes.append(outcome)
            print_attempt_summary(outcome)
            if outcome.row["status"] != "oom":
                break
            if index == len(attempts) - 1:
                break
        if args.fail_fast and outcomes[-1].row["status"] == "error":
            break

    rows = [outcome.row for outcome in outcomes]
    write_csv(peaks_path, rows, args.append)
    overall_status = "completed"
    if any(row["status"] == "error" for row in rows):
        overall_status = "completed_with_errors"
    elif any(row["status"] == "oom" for row in rows):
        overall_status = "completed_with_oom"
    write_json(metadata_path, make_metadata(args, overall_status, environment, outcomes))

    snapshot_count = sum(outcome.local_snapshot is not None for outcome in outcomes)
    trace_count = sum(outcome.local_profiler_trace is not None for outcome in outcomes)
    print(f"已写入轻量汇总：{peaks_path}、{metadata_path}")
    print(f"本地未提交原始文件：{snapshot_count} 个 snapshot、{trace_count} 个 profiler trace；目录：{args.snapshot_dir}")
    return 1 if any(row["status"] == "error" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
