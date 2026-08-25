#!/usr/bin/env python3
"""Reproducible end-to-end benchmarking and ``torch.profiler`` tracing.

Examples
--------
Run the required small-model FP32 benchmark on a GPU::

    python profiling/benchmark.py --model-size small --batch-size 4 \
      --context-length 512 --mode train_step --warmup 5 --steps 10 \
      --dtype fp32 --output results/benchmark.csv

Capture the six required complete-training-step traces::

    python profiling/benchmark.py --profile-matrix --profile-model-sizes small medium \
      --profile-context-lengths 256 512 1024 --trace-dir results/profile

Profiler traces are intentionally written locally and are not intended for a
submission repository.  The generated CSV and JSON metadata are lightweight
summaries suitable for copying into the separate assignment submission.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import shlex
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
try:
    from .nvtx_ranges import patched_attention_ranges, profile_range
except ImportError:
    from nvtx_ranges import patched_attention_ranges, profile_range


Mode = Literal["forward", "forward_backward", "train_step"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """The fixed language-model dimensions from Table 1 of the handout."""

    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig("small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig("medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig("large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig("xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
MODES: tuple[Mode, ...] = ("forward", "forward_backward", "train_step")
DTYPE_NAMES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def utc_timestamp() -> str:
    """Return an ISO-8601 timestamp with a UTC suffix."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Normalize the public ``fp32``/``bf16`` dtype interface."""

    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError(f"Unsupported compute dtype {dtype}; use torch.float32 or torch.bfloat16.")
        return dtype
    try:
        return DTYPE_NAMES[dtype.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(DTYPE_NAMES))
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {choices}.") from error


def dtype_name(dtype: str | torch.dtype) -> str:
    """Return the stable metadata spelling for a supported dtype."""

    return "bf16" if parse_dtype(dtype) is torch.bfloat16 else "fp32"


def resolve_device(device: str | torch.device) -> torch.device:
    """Validate a requested device and give a useful CUDA-unavailable error."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Run on a CUDA-enabled PyTorch/GPU environment, "
            "or use --device cpu only for a functional smoke test."
        )
    if resolved.type not in {"cuda", "cpu"}:
        raise ValueError(f"Only CUDA and CPU devices are supported, received {resolved!s}.")
    return resolved


def synchronize(device: str | torch.device) -> None:
    """Synchronize the host with CUDA, and do nothing for CPU execution."""

    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def set_seed(seed: int, device: str | torch.device) -> None:
    """Seed model construction and generated tokens before timing begins."""

    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)


def model_config(
    model_size: str,
    *,
    d_model: int | None = None,
    d_ff: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
) -> ModelConfig:
    """Load a named Table-1 configuration and apply explicit smoke-test overrides."""

    try:
        config = MODEL_CONFIGS[model_size]
    except KeyError as error:
        raise ValueError(f"Unknown model size {model_size!r}; choose one of {', '.join(MODEL_CONFIGS)}.") from error

    config = replace(
        config,
        d_model=config.d_model if d_model is None else d_model,
        d_ff=config.d_ff if d_ff is None else d_ff,
        num_layers=config.num_layers if num_layers is None else num_layers,
        num_heads=config.num_heads if num_heads is None else num_heads,
    )
    if min(config.d_model, config.d_ff, config.num_layers, config.num_heads) <= 0:
        raise ValueError("Model dimensions must all be positive.")
    if config.d_model % config.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if (config.d_model // config.num_heads) % 2 != 0:
        raise ValueError("d_model / num_heads must be even because the reference model uses RoPE.")
    return config


def build_model(
    model_size: str | ModelConfig,
    context_length: int,
    vocab_size: int = 10_000,
    device: str | torch.device = "cuda",
    *,
    d_model: int | None = None,
    d_ff: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
) -> BasicsTransformerLM:
    """Construct an FP32 basics Transformer outside the measurement interval."""

    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one.")
    config = (
        model_size
        if isinstance(model_size, ModelConfig)
        else model_config(
            model_size,
            d_model=d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    )
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    ).to(resolve_device(device))


def make_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate fixed random token IDs and targets before warm-up or timing."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one.")
    resolved = resolve_device(device)
    shape = (batch_size, context_length)
    return (
        torch.randint(vocab_size, shape, device=resolved, dtype=torch.long),
        torch.randint(vocab_size, shape, device=resolved, dtype=torch.long),
    )


def create_optimizer(model: torch.nn.Module, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
    """Use the assignment's AdamW implementation for full train-step runs."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    return AdamW(model.parameters(), lr=learning_rate)


def autocast_context(device: str | torch.device, dtype: str | torch.dtype):
    """Return a no-op FP32 context or BF16 autocast for the selected backend."""

    resolved_dtype = parse_dtype(dtype)
    if resolved_dtype is torch.float32:
        return nullcontext()
    resolved_device = torch.device(device)
    return torch.autocast(device_type=resolved_device.type, dtype=resolved_dtype)


def execute_workload(
    mode: Mode | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: str | torch.device,
    dtype: str | torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Run exactly one selected workload with required phase annotations.

    ``forward`` includes no loss, gradients, or optimizer work.  The other two
    modes include forward, cross-entropy loss, and backward.  ``train_step``
    additionally includes both gradient clearing and ``optimizer.step()`` in
    the optimizer range.  Gradients are cleared before every non-forward step,
    so ``forward_backward`` never accumulates across measurements.
    """

    normalized_mode = cast(Mode, mode)
    if normalized_mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose one of {', '.join(MODES)}.")
    if normalized_mode != "forward" and optimizer is None:
        raise ValueError(f"mode={normalized_mode!r} requires an optimizer for gradient clearing.")

    if normalized_mode == "forward":
        model.eval()
        with profile_range("forward"):
            with torch.no_grad(), autocast_context(device, dtype):
                return model(input_ids)

    model.train()
    assert optimizer is not None
    if normalized_mode == "train_step":
        with profile_range("optimizer"):
            optimizer.zero_grad(set_to_none=True)
    else:
        # This is intentionally outside the named forward/backward phases.  It
        # prevents cross-step accumulation without claiming that clearing grads
        # is part of a forward+backward-only measurement.
        optimizer.zero_grad(set_to_none=True)

    with profile_range("forward"):
        with autocast_context(device, dtype):
            logits = model(input_ids)
            loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten())

    with profile_range("backward"):
        loss.backward()

    if normalized_mode == "train_step":
        with profile_range("optimizer"):
            optimizer.step()
    return loss.detach()


def run_warmup(
    *,
    count: int,
    mode: Mode,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Run unmeasured warm-up work and synchronize after every CUDA step."""

    for _ in range(count):
        with profile_range("profile/warmup"):
            execute_workload(mode, model, optimizer, input_ids, targets, device, dtype)
            synchronize(device)


def time_measurement_step(
    *,
    mode: Mode,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Time one measurement step, bracketing CUDA work with synchronization."""

    synchronize(device)
    started = time.perf_counter()
    with profile_range("profile/measure"):
        execute_workload(mode, model, optimizer, input_ids, targets, device, dtype)
        synchronize(device)
    return (time.perf_counter() - started) * 1_000.0


def timing_statistics(timings_ms: Sequence[float]) -> dict[str, float]:
    """Compute mean, sample standard deviation, and coefficient of variation."""

    if not timings_ms:
        raise ValueError("At least one measurement timing is required.")
    mean_ms = statistics.fmean(timings_ms)
    std_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "cv_percent": 0.0 if mean_ms == 0 else 100.0 * std_ms / mean_ms,
    }


def parameter_count(model: torch.nn.Module) -> int:
    """Count all model parameters, including embeddings and tied parameters once."""

    return sum(parameter.numel() for parameter in model.parameters())


def safe_run_name(name: str) -> str:
    """Make a caller-provided trace name safe for use as a single filename."""

    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not result:
        raise ValueError("Trace name must contain at least one letter or number.")
    return result


def redact_argument(value: str) -> str:
    """Avoid accidentally placing absolute local paths in shareable metadata."""

    try:
        path = Path(value)
    except (TypeError, ValueError):
        return value
    if path.is_absolute():
        return f"<absolute-path>/{path.name}" if path.name else "<absolute-path>"
    return value


def recorded_command(argv: Sequence[str]) -> str:
    """Return a reproducible command while redacting absolute local paths."""

    return shlex.join(["python", "profiling/benchmark.py", *(redact_argument(arg) for arg in argv)])


def result_reference(path: Path) -> str:
    """Use a relative result path when possible, never the current absolute path."""

    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def environment_metadata(device: torch.device) -> dict[str, Any]:
    """Collect public software and accelerator information without host identity."""

    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device_type": device.type,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_capability": list(torch.cuda.get_device_capability(index)),
                "cuda_device_count": torch.cuda.device_count(),
            }
        )
    return metadata


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, human-readable JSON and create its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")


def write_benchmark_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Persist every raw timing with repeated summary fields for easy filtering."""

    fields = [
        "run_name",
        "stage",
        "timestamp_utc",
        "mode",
        "model_size",
        "batch_size",
        "context_length",
        "vocab_size",
        "dtype",
        "device",
        "warmup_steps",
        "measurement_steps",
        "seed",
        "parameter_count",
        "measurement_index",
        "elapsed_ms",
        "mean_ms",
        "std_ms",
        "cv_percent",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def benchmark_run(
    *,
    config: ModelConfig,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    mode: Mode,
    warmup: int,
    steps: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    learning_rate: float,
) -> tuple[list[float], int]:
    """Build once, warm up, and return raw timed samples plus parameter count."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    set_seed(seed, device)
    model = build_model(config, context_length, vocab_size, device)
    optimizer = create_optimizer(model, learning_rate)
    input_ids, targets = make_random_batch(batch_size, context_length, vocab_size, device)
    synchronize(device)  # Model/data initialization is never part of the timing interval.
    run_warmup(
        count=warmup,
        mode=mode,
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        device=device,
        dtype=dtype,
    )
    timings_ms = [
        time_measurement_step(
            mode=mode,
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            device=device,
            dtype=dtype,
        )
        for _ in range(steps)
    ]
    return timings_ms, parameter_count(model)


def run_benchmark(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Execute an end-to-end benchmark and write raw timings plus metadata."""

    config = config_from_args(args)
    timings_ms, num_parameters = benchmark_run(
        config=config,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        mode=cast(Mode, args.mode),
        warmup=args.warmup,
        steps=args.steps,
        dtype=dtype,
        seed=args.seed,
        device=device,
        learning_rate=args.learning_rate,
    )
    stats = timing_statistics(timings_ms)
    run_name = safe_run_name(f"{config.name}_{args.mode}_ctx{args.context_length}_{dtype_name(dtype)}")
    timestamp = utc_timestamp()
    rows = [
        {
            "run_name": run_name,
            "timestamp_utc": timestamp,
            "mode": args.mode,
            "model_size": config.name,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "dtype": dtype_name(dtype),
            "device": str(device),
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "seed": args.seed,
            "parameter_count": num_parameters,
            "measurement_index": index,
            "elapsed_ms": elapsed_ms,
            **stats,
        }
        for index, elapsed_ms in enumerate(timings_ms)
    ]
    output_path = Path(args.output)
    write_benchmark_csv(output_path, rows)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "schema_version": 1,
        "timestamp_utc": timestamp,
        "command": recorded_command(args.command_argv),
        "mode": args.mode,
        "model": asdict(config),
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "dtype": dtype_name(dtype),
        "seed": args.seed,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "parameter_count": num_parameters,
        "timing_unit": "milliseconds",
        "synchronization": "torch.cuda.synchronize() before and after each CUDA measurement step",
        "raw_timing_path": result_reference(output_path),
        "statistics": stats,
        "environment": environment_metadata(device),
    }
    write_json(metadata_path, metadata)
    return {
        "run_name": run_name,
        "mean_ms": stats["mean_ms"],
        "std_ms": stats["std_ms"],
        "cv_percent": stats["cv_percent"],
        "benchmark_csv": result_reference(output_path),
        "metadata": result_reference(metadata_path),
    }


def profiler_activities(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    """Enable CPU everywhere and CUDA activities whenever a CUDA trace is requested."""

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def profiler_summary_rows(profiler: torch.profiler.profile, run_name: str) -> list[dict[str, Any]]:
    """Convert the profiler operator table to portable, machine-readable rows."""

    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        rows.append(
            {
                "run_name": run_name,
                "stage": event.key if str(event.key).startswith(("profile/", "attention/")) or event.key in {"forward", "backward", "optimizer"} else "operator_or_kernel",
                "op_name": event.key,
                "calls": event.count,
                "cpu_self_time_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "cpu_total_time_us": float(getattr(event, "cpu_time_total", 0.0)),
                "cuda_self_time_us": float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)) or 0.0),
                "cuda_total_time_us": float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0)) or 0.0),
            }
        )
    return rows


def write_trace_summary(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write lightweight CPU/CUDA time and call-count summary for trace runs."""

    fields = [
        "run_name",
        "stage",
        "op_name",
        "calls",
        "cpu_self_time_us",
        "cpu_total_time_us",
        "cuda_self_time_us",
        "cuda_total_time_us",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def profile_one_train_step(
    *,
    config: ModelConfig,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    warmup: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
    learning_rate: float,
    trace_dir: Path,
    run_name: str,
    command_argv: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Capture one stable complete train step after normal and profiler warm-up.

    The ``wait=0, warmup=1, active=1`` schedule makes the internal warm-up
    step unrecorded while retaining one measurement step in the Chrome trace.
    Both steps still carry NVTX/record-function phase labels for Nsight and
    framework-level inspection respectively.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative.")
    trace_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed, device)
    model = build_model(config, context_length, vocab_size, device)
    optimizer = create_optimizer(model, learning_rate)
    input_ids, targets = make_random_batch(batch_size, context_length, vocab_size, device)
    synchronize(device)
    run_warmup(
        count=warmup,
        mode="train_step",
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        device=device,
        dtype=dtype,
    )

    schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    with torch.profiler.profile(
        activities=profiler_activities(device),
        schedule=schedule,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with profile_range("profile/warmup"):
            execute_workload("train_step", model, optimizer, input_ids, targets, device, dtype)
            synchronize(device)
        profiler.step()

        synchronize(device)
        started = time.perf_counter()
        with profile_range("profile/measure"):
            execute_workload("train_step", model, optimizer, input_ids, targets, device, dtype)
            synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        profiler.step()

    trace_path = trace_dir / f"{run_name}.json"
    profiler.export_chrome_trace(str(trace_path))
    summary_rows = profiler_summary_rows(profiler, run_name)
    metadata = {
        "schema_version": 1,
        "timestamp_utc": utc_timestamp(),
        "command": recorded_command(command_argv),
        "tool": "torch.profiler",
        "mode": "train_step",
        "model": asdict(config),
        "batch_size": batch_size,
        "context_length": context_length,
        "vocab_size": vocab_size,
        "dtype": dtype_name(dtype),
        "seed": seed,
        "normal_warmup_steps": warmup,
        "profiler_schedule": {"wait": 0, "warmup": 1, "active": 1, "repeat": 1},
        "profiled_measurement_steps": 1,
        "measurement_elapsed_ms": elapsed_ms,
        "parameter_count": parameter_count(model),
        "trace_file": trace_path.name,
        "stage_ranges": [
            "profile/warmup",
            "profile/measure",
            "forward",
            "backward",
            "optimizer",
            "attention",
            "attention/scores",
            "attention/softmax",
            "attention/value",
        ],
        "environment": environment_metadata(device),
    }
    return summary_rows, metadata


def run_profile(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Capture one trace for the selected configuration and write its summaries."""

    config = config_from_args(args)
    run_name = safe_run_name(args.trace_name or f"{config.name}_ctx{args.context_length}_train_step_{dtype_name(dtype)}")
    trace_dir = Path(args.trace_dir)
    with patched_attention_ranges():
        summary_rows, metadata = profile_one_train_step(
            config=config,
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            warmup=args.warmup,
            dtype=dtype,
            seed=args.seed,
            device=device,
            learning_rate=args.learning_rate,
            trace_dir=trace_dir,
            run_name=run_name,
            command_argv=args.command_argv,
        )
    summary_path = trace_dir / "trace_summary.csv"
    write_trace_summary(summary_path, summary_rows)
    metadata_path = trace_dir / "run_metadata.json"
    write_json(metadata_path, metadata)
    return {
        "run_name": run_name,
        "trace": result_reference(trace_dir / f"{run_name}.json"),
        "trace_summary": result_reference(summary_path),
        "metadata": result_reference(metadata_path),
    }


def run_profile_matrix(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    """Capture the required two-model by three-context six-trace matrix."""

    sizes = list(args.profile_model_sizes)
    contexts = list(args.profile_context_lengths)
    if len(sizes) != 2 or len(contexts) != 3:
        raise ValueError("--profile-matrix requires exactly two model sizes and three context lengths.")
    if len(set(sizes)) != 2:
        raise ValueError("--profile-model-sizes must name two distinct model sizes.")
    if len(set(contexts)) != 3 or any(context <= 128 or context & (context - 1) for context in contexts):
        raise ValueError("Profile contexts must be three distinct powers of two greater than 128.")

    trace_dir = Path(args.trace_dir)
    all_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    with patched_attention_ranges():
        for size in sizes:
            config = model_config(size)
            for context_length in contexts:
                run_name = safe_run_name(f"{size}_ctx{context_length}_train_step_{dtype_name(dtype)}")
                rows, metadata = profile_one_train_step(
                    config=config,
                    batch_size=args.batch_size,
                    context_length=context_length,
                    vocab_size=args.vocab_size,
                    warmup=args.warmup,
                    dtype=dtype,
                    seed=args.seed,
                    device=device,
                    learning_rate=args.learning_rate,
                    trace_dir=trace_dir,
                    run_name=run_name,
                    command_argv=args.command_argv,
                )
                all_rows.extend(rows)
                run_metadata.append(metadata)

    summary_path = trace_dir / "trace_summary.csv"
    metadata_path = trace_dir / "run_metadata.json"
    write_trace_summary(summary_path, all_rows)
    write_json(
        metadata_path,
        {
            "schema_version": 1,
            "timestamp_utc": utc_timestamp(),
            "tool": "torch.profiler",
            "trace_count": len(run_metadata),
            "runs": run_metadata,
        },
    )
    return {
        "trace_count": len(run_metadata),
        "trace_dir": result_reference(trace_dir),
        "trace_summary": result_reference(summary_path),
        "metadata": result_reference(metadata_path),
    }


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    """Translate parser overrides into the configuration used by a normal run."""

    return model_config(
        args.model_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser without executing any CUDA initialization."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="small", help="Table-1 model size (default: small).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument("--context-length", type=int, default=512, help="Token context length (default: 512).")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size (default: 10000).")
    parser.add_argument("--mode", choices=MODES, default="train_step", help="Workload to benchmark.")
    parser.add_argument("--warmup", type=int, default=5, help="Unmeasured warm-up steps (default: 5).")
    parser.add_argument("--steps", type=int, default=10, help="Measured benchmark steps (default: 10).")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32", help="Autocast compute dtype (default: fp32).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model and fixed random batch.")
    parser.add_argument("--device", default="cuda", help="CUDA device (default: cuda); use cpu only for smoke tests.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate for train modes.")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"), help="Raw timing CSV output path.")

    smoke = parser.add_argument_group("small CPU smoke-test overrides")
    smoke.add_argument("--d-model", type=int, help="Override d_model; not a Table-1 experiment configuration.")
    smoke.add_argument("--d-ff", type=int, help="Override d_ff; not a Table-1 experiment configuration.")
    smoke.add_argument("--num-layers", type=int, help="Override layer count; not a Table-1 experiment configuration.")
    smoke.add_argument("--num-heads", type=int, help="Override head count; not a Table-1 experiment configuration.")

    profiling = parser.add_argument_group("torch.profiler tracing")
    profile_target = profiling.add_mutually_exclusive_group()
    profile_target.add_argument("--profile", action="store_true", help="Capture one complete train-step Chrome trace.")
    profile_target.add_argument("--profile-matrix", action="store_true", help="Capture the required 2 x 3 = 6 train-step traces.")
    profiling.add_argument("--trace-dir", type=Path, default=Path("results/profile"), help="Local directory for Chrome traces and summaries.")
    profiling.add_argument("--trace-name", help="Filename stem for --profile (default is derived from the configuration).")
    profiling.add_argument(
        "--profile-model-sizes",
        nargs=2,
        choices=tuple(MODEL_CONFIGS),
        default=("small", "medium"),
        metavar=("SIZE_A", "SIZE_B"),
        help="Two sizes for --profile-matrix (default: small medium).",
    )
    profiling.add_argument(
        "--profile-context-lengths",
        nargs=3,
        type=int,
        default=(256, 512, 1024),
        metavar=("CTX_A", "CTX_B", "CTX_C"),
        help="Three powers of two greater than 128 for --profile-matrix.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint. Return non-zero with an actionable message on invalid setup."""

    parser = build_parser()
    command_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(command_argv)
    args.command_argv = command_argv
    try:
        device = resolve_device(args.device)
        dtype = parse_dtype(args.dtype)
        if args.profile_matrix:
            result = run_profile_matrix(args, device, dtype)
        elif args.profile:
            result = run_profile(args, device, dtype)
        else:
            result = run_benchmark(args, device, dtype)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
