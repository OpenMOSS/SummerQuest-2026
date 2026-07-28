"""Unified CUDA benchmark, profiler, and memory-capture entry point for A2-P.

Examples:
    uv run python profiling/benchmark.py --model-size small --mode train_step \
        --warmup 5 --steps 10 --dtype fp32 --output results/benchmark/raw.jsonl

    uv run python profiling/benchmark.py --model-size xl --context-length 2048 \
        --mode train_step --warmup 5 --steps 1 --track-memory \
        --memory-snapshot local_artifacts/memory/xl_2048_train_step.pickle
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import statistics
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW, get_cosine_lr
from profiling.collect_utils import is_cuda_oom_text, requested_allocation_bytes
from profiling.memory_snapshot import MemorySnapshot, memory_metadata, memory_statistics
from profiling.nvtx_ranges import install_attention_ranges, phase
from profiling.trace_summary import write_trace_summary


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSpec(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSpec(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B": ModelSpec(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


class Mode(StrEnum):
    FORWARD = "forward"
    FORWARD_BACKWARD = "forward_backward"
    TRAIN_STEP = "train_step"


class Precision(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    batch_size: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class RunConfig:
    model_size: str
    mode: Mode
    precision: Precision
    warmup_steps: int
    measurement_steps: int
    device: str
    seed: int
    lr_max: float
    lr_min: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    grad_clip: float
    nvtx: bool
    profile_tool: str
    track_memory: bool


@dataclass
class FailureContext:
    """Track the public execution boundary active when a CUDA OOM is raised."""

    scope: str = "initialization"
    phase: str | None = None
    device: torch.device | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    model_config: ModelConfig
    run_config: RunConfig
    parameter_count: int
    raw_timings_ms: list[float]
    mean_ms: float
    std_ms: float
    cv: float
    memory: dict[str, Any] | None
    environment: dict[str, str | None]
    timestamp_utc: str
    command: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "timestamp_utc": self.timestamp_utc,
            "timer": "torch.cuda.Event.elapsed_time",
            "unit": "milliseconds",
            "command": self.command,
            "model_config": asdict(self.model_config),
            "run_config": {**asdict(self.run_config), "mode": self.run_config.mode.value, "precision": self.run_config.precision.value},
            "parameter_count": self.parameter_count,
            "raw_timings_ms": self.raw_timings_ms,
            "statistics": {"mean_ms": self.mean_ms, "std_ms": self.std_ms, "cv": self.cv, "n": len(self.raw_timings_ms)},
            "memory": self.memory,
            "environment": self.environment,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure or profile one CUDA Transformer workload.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), default="small")
    parser.add_argument("--mode", choices=[mode.value for mode in Mode], default=Mode.TRAIN_STEP.value)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=[precision.value for precision in Precision], default=Precision.FP32.value)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr-max", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for nsys.")
    parser.add_argument("--profile-tool", choices=("none", "torch"), default="none")
    parser.add_argument("--trace-output", type=Path, default=None, help="Chrome trace path when --profile-tool torch is used.")
    parser.add_argument("--profile-summary", type=Path, default=None, help="Compact operator summary CSV for torch.profiler.")
    parser.add_argument("--track-memory", action="store_true", help="Record active/allocated/reserved peak statistics.")
    parser.add_argument("--memory-snapshot", type=Path, default=None, help="PyTorch memory-history pickle path; do not commit it.")
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=None,
        help="Write a sanitized structured CUDA-OOM telemetry JSON record before exiting non-zero.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Append the complete result record as JSONL.")
    parser.add_argument("--metadata", type=Path, default=None, help="Write this run's public, lightweight metadata JSON.")
    return parser


def resolve_configs(args: argparse.Namespace) -> tuple[ModelConfig, RunConfig]:
    spec = MODEL_SPECS[args.model_size]
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        d_model=spec.d_model if args.d_model is None else args.d_model,
        d_ff=spec.d_ff if args.d_ff is None else args.d_ff,
        num_layers=spec.num_layers if args.num_layers is None else args.num_layers,
        num_heads=spec.num_heads if args.num_heads is None else args.num_heads,
    )
    run_config = RunConfig(
        model_size=args.model_size,
        mode=Mode(args.mode),
        precision=Precision(args.dtype),
        warmup_steps=args.warmup,
        measurement_steps=args.steps,
        device=args.device,
        seed=args.seed,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        grad_clip=args.grad_clip,
        nvtx=args.nvtx,
        profile_tool=args.profile_tool,
        track_memory=args.track_memory or args.memory_snapshot is not None,
    )
    return model_config, run_config


def validate_configs(model: ModelConfig, run: RunConfig) -> torch.device:
    positive = {
        "vocab_size": model.vocab_size,
        "context_length": model.context_length,
        "batch_size": model.batch_size,
        "d_model": model.d_model,
        "d_ff": model.d_ff,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "steps": run.measurement_steps,
    }
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}.")
    if run.warmup_steps < 0:
        raise ValueError("warmup must be non-negative.")
    if model.d_model % model.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if not 0.0 <= run.lr_min <= run.lr_max:
        raise ValueError("Require 0 <= lr_min <= lr_max.")
    if not 0.0 <= run.beta1 < 1.0 or not 0.0 <= run.beta2 < 1.0:
        raise ValueError("beta1 and beta2 must be in [0, 1).")
    if run.eps < 0.0 or run.weight_decay < 0.0 or run.grad_clip <= 0.0:
        raise ValueError("eps/weight_decay/grad_clip must be valid non-negative values; grad_clip must be positive.")
    device = torch.device(run.device)
    if device.type != "cuda":
        raise ValueError("A2-P benchmark requires a CUDA device.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available in this PyTorch environment.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested {device}, but only {torch.cuda.device_count()} CUDA device(s) are available.")
    if run.precision is Precision.BF16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"{device} does not support CUDA BF16 autocast.")
    return torch.device("cuda", torch.cuda.current_device()) if device.index is None else device


def validate_auxiliary_args(args: argparse.Namespace) -> None:
    """Reject incomplete profiler configuration before allocating a model."""

    if args.profile_tool == "torch" and (args.trace_output is None or args.profile_summary is None):
        raise ValueError("--profile-tool torch requires both --trace-output and --profile-summary.")


def build_model(config: ModelConfig) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )


def random_batch(config: ModelConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (config.batch_size, config.context_length)
    return (
        torch.randint(config.vocab_size, shape, device=device, dtype=torch.long),
        torch.randint(config.vocab_size, shape, device=device, dtype=torch.long),
    )


_OOM_MEMORY_STAT_KEYS = {
    "active_bytes": "active_bytes.all.current",
    "peak_active_bytes": "active_bytes.all.peak",
}


def _safe_cuda_int(operation: Any) -> int | None:
    """Run best-effort CUDA telemetry without allowing it to mask an OOM."""

    try:
        return int(operation())
    except Exception:
        return None


def _failure_memory_telemetry(device: torch.device | None, error_text: str) -> dict[str, Any]:
    """Capture only numeric allocator state that remains meaningful after CUDA OOM."""

    stats: dict[str, Any] | None
    try:
        stats = torch.cuda.memory_stats(device) if device is not None else None
    except Exception:
        stats = None

    values = {
        name: (int(stats[key]) if stats is not None and key in stats else None)
        for name, key in _OOM_MEMORY_STAT_KEYS.items()
    }
    values.update(
        {
            "allocated_bytes": _safe_cuda_int(lambda: torch.cuda.memory_allocated(device)) if device is not None else None,
            "peak_allocated_bytes": _safe_cuda_int(lambda: torch.cuda.max_memory_allocated(device)) if device is not None else None,
            "reserved_bytes": _safe_cuda_int(lambda: torch.cuda.memory_reserved(device)) if device is not None else None,
            "peak_reserved_bytes": _safe_cuda_int(lambda: torch.cuda.max_memory_reserved(device)) if device is not None else None,
        }
    )
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device) if device is not None else (None, None)
    except Exception:
        free_bytes, total_bytes = None, None
    values.update(
        {
            "free_bytes": int(free_bytes) if free_bytes is not None else None,
            "total_bytes": int(total_bytes) if total_bytes is not None else None,
            "requested_allocation_bytes": requested_allocation_bytes(error_text),
        }
    )
    available_fields = [name for name, value in values.items() if value is not None]
    return {
        "telemetry_status": "available" if len(available_fields) == len(values) else "partial" if available_fields else "unavailable",
        "unavailable_fields": [name for name, value in values.items() if value is None],
        "statistics_bytes": {
            name: values[name]
            for name in (
                "active_bytes",
                "peak_active_bytes",
                "allocated_bytes",
                "peak_allocated_bytes",
                "reserved_bytes",
                "peak_reserved_bytes",
            )
        },
        "free_bytes": values["free_bytes"],
        "total_bytes": values["total_bytes"],
        "requested_allocation_bytes": values["requested_allocation_bytes"],
    }


def _failure_environment(device: torch.device | None) -> dict[str, str | None]:
    """Collect public environment fields while retaining an explicit unavailable value."""

    device_name: str | None
    try:
        device_name = torch.cuda.get_device_name(device) if device is not None else None
    except Exception:
        device_name = None
    return {
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def failure_telemetry(
    *,
    model_config: ModelConfig,
    run_config: RunConfig,
    context: FailureContext,
    device: torch.device | None,
    error_text: str,
) -> dict[str, Any]:
    """Build the sanitized, child-process-only record used to repair OOM results."""

    peak_scope = {
        "initialization": "initialization",
        "warmup": "warmup",
        "measurement": "post_warmup_measurement",
    }[context.scope]
    run_record = asdict(run_config)
    run_record["mode"] = run_config.mode.value
    run_record["precision"] = run_config.precision.value
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "exception": "cuda_oom",
        "model_config": asdict(model_config),
        "run_config": run_record,
        "failure_scope": context.scope,
        "failure_phase": context.phase,
        "peak_scope": peak_scope,
        "memory": _failure_memory_telemetry(device, error_text),
        "environment": _failure_environment(device),
    }


def write_failure_telemetry(path: Path, telemetry: dict[str, Any]) -> None:
    """Atomically publish OOM telemetry without retaining the exception text itself."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(telemetry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def autocast_context(precision: Precision):
    if precision is Precision.BF16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def execute_step(
    *,
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    x: torch.Tensor,
    y: torch.Tensor,
    run: RunConfig,
    global_step: int,
    nvtx: bool,
    record_function: bool,
    failure_context: FailureContext | None = None,
) -> None:
    """Run one mode with explicit stage boundaries and no timing logic."""

    if run.mode is Mode.FORWARD:
        model.eval()
        if failure_context is not None:
            failure_context.phase = "forward"
        with torch.no_grad(), phase("forward", nvtx=nvtx, record_function=record_function), autocast_context(run.precision):
            model(x)
        return

    assert optimizer is not None
    model.train()
    if failure_context is not None:
        failure_context.phase = "optimizer"
    optimizer.zero_grad(set_to_none=True)
    if failure_context is not None:
        failure_context.phase = "forward"
    with phase("forward", nvtx=nvtx, record_function=record_function), autocast_context(run.precision):
        logits = model(x)
        loss = cross_entropy(logits, y)
    if failure_context is not None:
        failure_context.phase = "backward"
    with phase("backward", nvtx=nvtx, record_function=record_function):
        loss.backward()
    if run.mode is Mode.FORWARD_BACKWARD:
        return

    if failure_context is not None:
        failure_context.phase = "optimizer"
    with phase("optimizer", nvtx=nvtx, record_function=record_function):
        clip_gradient(model.parameters(), run.grad_clip)
        learning_rate = get_cosine_lr(
            global_step + 1,
            run.lr_max,
            run.lr_min,
            run.warmup_steps,
            run.warmup_steps + run.measurement_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.step()


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


class CudaEventTimer:
    """Reuse one event pair so event construction cannot perturb a measurement."""

    def __init__(self) -> None:
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def measure(self, *, stream: torch.cuda.Stream, execute: Any, device: torch.device) -> float:
        self.start_event.record(stream)
        execute()
        self.end_event.record(stream)
        synchronize(device)
        return self.start_event.elapsed_time(self.end_event)


def profiler_context(run: RunConfig):
    if run.profile_tool != "torch":
        return nullcontext(None)
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=run.track_memory,
        with_stack=True,
    )


def warmup_partition(run: RunConfig) -> tuple[int, int]:
    """Split warm-up so a torch trace contains one final stable warm-up step."""

    profiled_steps = 1 if run.profile_tool == "torch" and run.warmup_steps > 0 else 0
    return run.warmup_steps - profiled_steps, profiled_steps


def write_profile_summary(trace_path: Path, path: Path) -> None:
    """Preserve ``--profile-summary`` while reducing only profile/measure trace data."""

    write_trace_summary(trace_path, path)


def public_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def public_command(arguments: list[str]) -> str:
    """Keep result metadata reproducible without exposing absolute local paths."""

    visible_arguments = [public_path(Path(argument)) if Path(argument).is_absolute() else argument for argument in arguments]
    return " ".join(shlex.quote(argument) for argument in visible_arguments)


def benchmark(model_config: ModelConfig, run_config: RunConfig, args: argparse.Namespace) -> BenchmarkResult:
    """Run a benchmark and, on CUDA OOM, let the child persist safe telemetry first."""

    context = FailureContext()
    try:
        return _benchmark(model_config, run_config, args, context)
    except RuntimeError as error:
        if not is_cuda_oom_text(str(error)):
            raise
        failure_output = getattr(args, "failure_output", None)
        if failure_output is not None:
            try:
                write_failure_telemetry(
                    failure_output,
                    failure_telemetry(
                        model_config=model_config,
                        run_config=run_config,
                        context=context,
                        device=context.device,
                        error_text=str(error),
                    ),
                )
            except Exception:
                # Preserve the original OOM exit even if allocator telemetry is unavailable.
                pass
        raise


def _benchmark(
    model_config: ModelConfig,
    run_config: RunConfig,
    args: argparse.Namespace,
    failure_context: FailureContext,
) -> BenchmarkResult:
    device = validate_configs(model_config, run_config)
    failure_context.device = device
    torch.manual_seed(run_config.seed)
    torch.cuda.manual_seed_all(run_config.seed)

    marked = run_config.nvtx or run_config.profile_tool == "torch"
    if marked:
        install_attention_ranges(nvtx=run_config.nvtx, record_function=run_config.profile_tool == "torch")

    model = build_model(model_config).to(device)
    optimizer = None
    if run_config.mode is not Mode.FORWARD:
        optimizer = AdamW(
            model.parameters(),
            lr=run_config.lr_max,
            betas=(run_config.beta1, run_config.beta2),
            eps=run_config.eps,
            weight_decay=run_config.weight_decay,
        )
    x, y = random_batch(model_config, device)

    unprofiled_warmup_steps, profiled_warmup_steps = warmup_partition(run_config)
    failure_context.scope = "warmup"
    failure_context.phase = None
    capture_failure_peaks = run_config.track_memory or getattr(args, "failure_output", None) is not None
    if capture_failure_peaks and device.type == "cuda":
        # A warm-up OOM should report warm-up-only peaks, not model/input setup peaks.
        torch.cuda.reset_peak_memory_stats(device)
    if unprofiled_warmup_steps:
        with phase("profile/warmup", nvtx=run_config.nvtx, record_function=False):
            for global_step in range(unprofiled_warmup_steps):
                execute_step(
                    model=model,
                    optimizer=optimizer,
                    x=x,
                    y=y,
                    run=run_config,
                    global_step=global_step,
                    nvtx=run_config.nvtx,
                    record_function=False,
                    failure_context=failure_context,
                )
                synchronize(device)
    # All warm-up work has synchronized; setup failures after this point have no stage attribution.
    failure_context.phase = None

    snapshot = MemorySnapshot(args.memory_snapshot)
    stream = torch.cuda.current_stream(device)
    timer = CudaEventTimer()
    raw_timings_ms: list[float] = []
    profiler = None
    try:
        with profiler_context(run_config) as profiler:
            if profiled_warmup_steps:
                with phase("profile/warmup", nvtx=run_config.nvtx, record_function=True):
                    global_step = unprofiled_warmup_steps
                    execute_step(
                        model=model,
                        optimizer=optimizer,
                        x=x,
                        y=y,
                        run=run_config,
                        global_step=global_step,
                        nvtx=run_config.nvtx,
                        record_function=True,
                        failure_context=failure_context,
                    )
                    synchronize(device)

            failure_context.scope = "measurement"
            failure_context.phase = None
            if capture_failure_peaks and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            snapshot.start()
            with phase("profile/measure", nvtx=run_config.nvtx, record_function=run_config.profile_tool == "torch"):
                for measurement_index in range(run_config.measurement_steps):
                    global_step = run_config.warmup_steps + measurement_index
                    raw_timings_ms.append(
                        timer.measure(
                            stream=stream,
                            device=device,
                            execute=lambda global_step=global_step: execute_step(
                                model=model,
                                optimizer=optimizer,
                                x=x,
                                y=y,
                                run=run_config,
                                global_step=global_step,
                                nvtx=run_config.nvtx,
                                record_function=run_config.profile_tool == "torch",
                                failure_context=failure_context,
                            ),
                        )
                    )
    finally:
        original_exception_is_active = sys.exc_info()[0] is not None
        try:
            snapshot.stop_and_dump()
        except Exception:
            if not original_exception_is_active:
                raise

    if profiler is not None:
        assert args.trace_output is not None
        assert args.profile_summary is not None
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.trace_output))
        write_profile_summary(args.trace_output, args.profile_summary)

    statistics_ms = statistics.mean(raw_timings_ms)
    std_ms = statistics.stdev(raw_timings_ms) if len(raw_timings_ms) > 1 else 0.0
    cv = std_ms / statistics_ms if statistics_ms else 0.0
    memory = memory_metadata(args.memory_snapshot, memory_statistics(device) if run_config.track_memory else None)
    environment = {
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
    }
    return BenchmarkResult(
        model_config=model_config,
        run_config=run_config,
        parameter_count=model.get_num_params(),
        raw_timings_ms=raw_timings_ms,
        mean_ms=statistics_ms,
        std_ms=std_ms,
        cv=cv,
        memory=memory,
        environment=environment,
        timestamp_utc=datetime.now(UTC).isoformat(),
        command=public_command(sys.argv),
    )


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True))
        output_file.write("\n")


def main(argv: list[str] | None = None) -> BenchmarkResult:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_auxiliary_args(args)
        model_config, run_config = resolve_configs(args)
        result = benchmark(model_config, run_config, args)
    except ValueError as error:
        parser.error(str(error))

    record = result.to_record()
    if args.output is not None:
        append_jsonl(args.output, record)
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
