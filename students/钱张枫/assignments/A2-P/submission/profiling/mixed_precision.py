#!/usr/bin/env python3
"""Run the mixed-precision experiments required by the profiling assignment.

The default invocation is intentionally a CUDA experiment.  It runs the four
FP16 accumulation cases from the handout, probes BF16 CUDA autocast on the
ToyModel, and compares FP32 with BF16 autocast for a language-model workload.
The JSON output contains only public hardware/software facts and experiment
configuration; it deliberately excludes host names, user names, UUIDs, and
local paths.

Examples:

    python profiling/mixed_precision.py --model-size small --mode train_step \\
        --output results/mixed_precision.json

    # A fast correctness-only path when CUDA is unavailable.
    python profiling/mixed_precision.py --cpu-test-mode --steps 1 --warmup 0 \\
        --output results/mixed_precision_cpu_smoke.json

    # The four accumulation cases do not require CUDA.
    python profiling/mixed_precision.py --accumulation-only \\
        --output results/mixed_precision_accumulation.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import tempfile
import time
from typing import Literal, TypeAlias, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from cs336_basics.model import BasicsTransformerLM
except ImportError as exc:  # pragma: no cover - gives a useful CLI failure in an incomplete environment.
    raise RuntimeError(
        "无法导入 cs336_basics.model.BasicsTransformerLM。请确认 assignment1-basics 依赖已按 pyproject.toml 安装。"
    ) from exc


BenchmarkMode: TypeAlias = Literal["forward", "forward_backward", "train_step"]
PrecisionName: TypeAlias = Literal["fp32", "bf16"]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class ModelConfig:
    """A language-model shape from Table 1, plus a deliberately tiny smoke shape."""

    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "smoke": ModelConfig("smoke", d_model=64, d_ff=256, num_layers=2, num_heads=4),
    "small": ModelConfig("small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig("medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig("large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig("xl", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig("10b", d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


@dataclass(frozen=True)
class ExperimentConfig:
    model_size: str
    batch_size: int
    context_length: int
    vocab_size: int
    warmup_steps: int
    measurement_steps: int
    mode: str
    seed: int
    learning_rate: float


@dataclass(frozen=True)
class RuntimeInfo:
    """Execution facts that are safe to place in a public result file."""

    device: torch.device
    is_cuda_experiment: bool
    execution_label: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TimingSummary:
    raw_step_ms: list[float]
    mean_ms: float
    sample_std_ms: float
    cv_percent: float


class ToyModel(nn.Module):
    """The LayerNorm-containing ToyModel from the fixed assignment handout."""

    def __init__(self, in_features: int = 10, out_features: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def dtype_name(dtype: torch.dtype) -> str:
    """Return PyTorch's stable, explicit dtype spelling for JSON output."""

    return str(dtype)


def finite_float(value: float) -> float | None:
    """JSON has no portable NaN/Infinity representation, so encode them as null."""

    return float(value) if math.isfinite(value) else None


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return finite_float(numerator / denominator)


def synchronize(device: torch.device) -> None:
    """Synchronize the CUDA stream; CPU execution is already synchronous."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, precision: PrecisionName) -> AbstractContextManager[None]:
    """Create an autocast context without falsely labelling FP32 as mixed precision."""

    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def resolve_runtime(device_argument: str, cpu_test_mode: bool) -> RuntimeInfo:
    """Resolve CUDA safely and make any CPU fallback explicit in the result."""

    try:
        requested_device = torch.device(device_argument)
    except RuntimeError as exc:
        raise ValueError(f"无效的 --device 值：{device_argument!r}") from exc

    if requested_device.type not in {"cuda", "cpu"}:
        raise ValueError("--device 只支持 cuda、cuda:N 或 cpu。")

    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            if not cpu_test_mode:
                raise RuntimeError(
                    "ToyModel BF16 autocast 和语言模型比较必须在 CUDA 上运行，但当前 PyTorch 未检测到可用 CUDA。"
                    "请在 CUDA 环境运行，或显式传入 --cpu-test-mode 执行不可用于性能结论的安全冒烟测试。"
                )
            return RuntimeInfo(
                device=torch.device("cpu"),
                is_cuda_experiment=False,
                execution_label="cpu_smoke_test_after_cuda_unavailable",
                warnings=("CUDA 不可用；结果仅用于 CPU 正确性冒烟，不能作为 CUDA BF16 性能或显存结论。",),
            )

        if requested_device.index is not None:
            torch.cuda.set_device(requested_device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16 autocast；请改用支持 BF16 的 NVIDIA GPU。")
        return RuntimeInfo(
            device=requested_device,
            is_cuda_experiment=True,
            execution_label="cuda_bf16_autocast",
            warnings=(),
        )

    if not cpu_test_mode:
        raise RuntimeError(
            "--device cpu 仅允许与 --cpu-test-mode 一起使用，以免将 CPU 结果误作 CUDA BF16 实验结论。"
        )
    return RuntimeInfo(
        device=requested_device,
        is_cuda_experiment=False,
        execution_label="cpu_smoke_test",
        warnings=("CPU autocast 是安全冒烟路径，不能与 CUDA 的时间或峰值显存结果比较。",),
    )


def run_accumulation_experiments() -> list[dict[str, JSONValue]]:
    """Run the four accumulation snippets exactly as specified in the handout."""

    expected = 10.0
    cases: list[tuple[str, torch.dtype, torch.dtype, bool]] = [
        ("fp32_accumulator_fp32_input", torch.float32, torch.float32, False),
        ("fp16_accumulator_fp16_input", torch.float16, torch.float16, False),
        ("fp32_accumulator_fp16_input", torch.float32, torch.float16, False),
        ("fp32_accumulator_explicit_fp16_to_fp32_input", torch.float32, torch.float16, True),
    ]
    results: list[dict[str, JSONValue]] = []

    for name, accumulator_dtype, input_dtype, explicit_upcast in cases:
        accumulator = torch.tensor(0, dtype=accumulator_dtype)
        for _ in range(1000):
            if explicit_upcast:
                # This is intentionally a separate temporary, matching the fourth handout snippet.
                value = torch.tensor(0.01, dtype=torch.float16)
                accumulator += value.type(torch.float32)
            else:
                accumulator += torch.tensor(0.01, dtype=input_dtype)
        result = float(accumulator.item())
        results.append(
            {
                "case": name,
                "iterations": 1000,
                "accumulator_dtype": dtype_name(accumulator_dtype),
                "input_dtype": dtype_name(input_dtype),
                "explicit_input_upcast_to_fp32": explicit_upcast,
                "result": finite_float(result),
                "expected_mathematical_sum": expected,
                "absolute_error": finite_float(abs(result - expected)),
            }
        )
    return results


def run_toymodel_dtype_probe(device: torch.device, seed: int) -> dict[str, JSONValue]:
    """Measure actual dtypes under BF16 autocast instead of assuming a policy outcome."""

    torch.manual_seed(seed)
    model = ToyModel().to(device)
    captured_dtypes: dict[str, str] = {}

    def capture_dtype(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            captured_dtypes[name] = dtype_name(output.dtype)

        return hook

    fc1_hook = model.fc1.register_forward_hook(capture_dtype("fc1_output"))
    layer_norm_hook = model.ln.register_forward_hook(capture_dtype("layer_norm_output"))
    try:
        inputs = torch.randn((8, 10), device=device, dtype=torch.float32)
        targets = torch.randint(0, 10, (8,), device=device)
        with autocast_context(device, "bf16"):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        gradient_dtypes = sorted(
            {dtype_name(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None}
        )
        parameter_dtypes = sorted({dtype_name(parameter.dtype) for parameter in model.parameters()})
        return {
            "autocast_dtype": dtype_name(torch.bfloat16),
            "parameter_dtypes": parameter_dtypes,
            "fc1_output_dtype": captured_dtypes.get("fc1_output"),
            "layer_norm_output_dtype": captured_dtypes.get("layer_norm_output"),
            "logits_dtype": dtype_name(logits.dtype),
            "loss_dtype": dtype_name(loss.dtype),
            "gradient_dtypes": gradient_dtypes,
            "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
        }
    finally:
        fc1_hook.remove()
        layer_norm_hook.remove()
        model.zero_grad(set_to_none=True)


def build_language_model(config: ModelConfig, vocab_size: int, context_length: int, device: torch.device, seed: int) -> nn.Module:
    """Create a fresh FP32 master-weight model for a fair precision comparison."""

    torch.manual_seed(seed)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
    return model.to(device)


def make_random_batch(
    batch_size: int, context_length: int, vocab_size: int, device: torch.device, seed: int
) -> tuple[Tensor, Tensor]:
    """Create one deterministic token batch outside every timed region."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    return input_ids, targets


def language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross entropy over all batch and context positions."""

    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def execute_workload(
    mode: BenchmarkMode,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    targets: Tensor,
    device: torch.device,
    precision: PrecisionName,
) -> None:
    """Run exactly one workload while keeping data generation outside the timing boundary."""

    if mode == "forward":
        with torch.no_grad(), autocast_context(device, precision):
            _ = model(input_ids)
        return

    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        logits = model(input_ids)
        loss = language_model_loss(logits, targets)
    loss.backward()
    if mode == "train_step":
        optimizer.step()


def numeric_snapshot(
    model: nn.Module, input_ids: Tensor, targets: Tensor, device: torch.device, precision: PrecisionName
) -> dict[str, JSONValue]:
    """Capture lightweight numerical signals before warm-up mutates train-step weights."""

    model.eval()
    with torch.no_grad(), autocast_context(device, precision):
        logits = model(input_ids)
        loss = language_model_loss(logits, targets)
    logits_fp32 = logits.detach().float()
    finite_logits = bool(torch.isfinite(logits_fp32).all().item())
    summary = {
        "logits_dtype": dtype_name(logits.dtype),
        "loss_dtype": dtype_name(loss.dtype),
        "loss": finite_float(float(loss.detach().float().item())),
        "loss_is_finite": bool(torch.isfinite(loss.detach()).item()),
        "logits_all_finite": finite_logits,
        "logits_mean": finite_float(float(logits_fp32.mean().item())),
        "logits_std_population": finite_float(float(logits_fp32.std(unbiased=False).item())),
        "logits_max_abs": finite_float(float(logits_fp32.abs().max().item())),
    }
    del logits_fp32, logits, loss
    model.train()
    return summary


def summarize_timings(raw_step_ms: list[float]) -> TimingSummary:
    if not raw_step_ms:
        raise ValueError("measurement_steps 必须大于 0。")
    mean_ms = statistics.fmean(raw_step_ms)
    sample_std_ms = statistics.stdev(raw_step_ms) if len(raw_step_ms) > 1 else 0.0
    cv_percent = 100.0 * sample_std_ms / mean_ms if mean_ms > 0.0 else 0.0
    return TimingSummary(
        raw_step_ms=[float(value) for value in raw_step_ms],
        mean_ms=float(mean_ms),
        sample_std_ms=float(sample_std_ms),
        cv_percent=float(cv_percent),
    )


def memory_snapshot(device: torch.device, baseline: bool = False) -> dict[str, JSONValue]:
    """Return a lightweight, clearly labelled CUDA memory reading."""

    if device.type != "cuda":
        return {
            "available": False,
            "reason": "CPU smoke tests do not report CUDA allocated/reserved/peak memory.",
        }
    prefix = "baseline_" if baseline else "peak_"
    return {
        "available": True,
        f"{prefix}allocated_bytes": int(
            torch.cuda.memory_allocated(device) if baseline else torch.cuda.max_memory_allocated(device)
        ),
        f"{prefix}reserved_bytes": int(
            torch.cuda.memory_reserved(device) if baseline else torch.cuda.max_memory_reserved(device)
        ),
    }


def run_precision_benchmark(
    precision: PrecisionName,
    mode: BenchmarkMode,
    model_config: ModelConfig,
    config: ExperimentConfig,
    runtime: RuntimeInfo,
    input_ids: Tensor,
    targets: Tensor,
) -> dict[str, JSONValue]:
    """Benchmark one fresh FP32 master-weight model under one autocast policy."""

    model = build_language_model(model_config, config.vocab_size, config.context_length, runtime.device, config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    try:
        numbers = numeric_snapshot(model, input_ids, targets, runtime.device, precision)

        for _ in range(config.warmup_steps):
            execute_workload(mode, model, optimizer, input_ids, targets, runtime.device, precision)
            synchronize(runtime.device)

        # Do not carry a final warm-up gradient into a forward/backward measurement.
        optimizer.zero_grad(set_to_none=True)
        synchronize(runtime.device)
        if runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        baseline_memory = memory_snapshot(runtime.device, baseline=True)

        raw_step_ms: list[float] = []
        for _ in range(config.measurement_steps):
            synchronize(runtime.device)
            start_time = time.perf_counter()
            execute_workload(mode, model, optimizer, input_ids, targets, runtime.device, precision)
            synchronize(runtime.device)
            raw_step_ms.append((time.perf_counter() - start_time) * 1_000.0)

        optimizer.zero_grad(set_to_none=True)
        synchronize(runtime.device)
        peak_memory = memory_snapshot(runtime.device)
        timing = summarize_timings(raw_step_ms)
        return {
            "precision": precision,
            "autocast_dtype": dtype_name(torch.bfloat16) if precision == "bf16" else None,
            "numeric_snapshot_before_warmup": numbers,
            "timing": asdict(timing),
            "memory": {
                "measurement_scope": "after warm-up; includes one model, optimizer state created by the selected workload, and measured steps",
                "baseline": baseline_memory,
                "peak": peak_memory,
            },
        }
    finally:
        del optimizer
        del model
        if runtime.device.type == "cuda":
            torch.cuda.empty_cache()


def float_difference(left: JSONValue, right: JSONValue) -> float | None:
    if not isinstance(left, (float, int)) or not isinstance(right, (float, int)):
        return None
    return finite_float(float(right) - float(left))


def make_precision_comparison(fp32: dict[str, JSONValue], bf16: dict[str, JSONValue]) -> dict[str, JSONValue]:
    """Summarize performance, memory, and numerical trends without retaining large logits."""

    fp32_timing = cast(dict[str, JSONValue], fp32["timing"])
    bf16_timing = cast(dict[str, JSONValue], bf16["timing"])
    fp32_numbers = cast(dict[str, JSONValue], fp32["numeric_snapshot_before_warmup"])
    bf16_numbers = cast(dict[str, JSONValue], bf16["numeric_snapshot_before_warmup"])
    fp32_memory = cast(dict[str, JSONValue], cast(dict[str, JSONValue], fp32["memory"])["peak"])
    bf16_memory = cast(dict[str, JSONValue], cast(dict[str, JSONValue], bf16["memory"])["peak"])

    fp32_mean = cast(float | None, fp32_timing["mean_ms"])
    bf16_mean = cast(float | None, bf16_timing["mean_ms"])
    speedup = safe_ratio(fp32_mean, bf16_mean)
    fp32_loss = fp32_numbers["loss"]
    bf16_loss = bf16_numbers["loss"]
    loss_delta = float_difference(fp32_loss, bf16_loss)
    loss_relative_delta = safe_ratio(loss_delta, cast(float | None, fp32_loss))

    fp32_peak = fp32_memory.get("peak_allocated_bytes")
    bf16_peak = bf16_memory.get("peak_allocated_bytes")
    memory_delta = float_difference(fp32_peak, bf16_peak)

    return {
        "bf16_over_fp32_speedup": speedup,
        "mean_step_time_delta_ms_bf16_minus_fp32": float_difference(fp32_mean, bf16_mean),
        "peak_allocated_memory_delta_bytes_bf16_minus_fp32": memory_delta,
        "bf16_over_fp32_peak_allocated_memory_ratio": safe_ratio(
            cast(float | None, bf16_peak), cast(float | None, fp32_peak)
        ),
        "numeric_trend_before_warmup": {
            "loss_delta_bf16_minus_fp32": loss_delta,
            "loss_relative_delta_bf16_minus_fp32": loss_relative_delta,
            "logits_mean_delta_bf16_minus_fp32": float_difference(
                fp32_numbers["logits_mean"], bf16_numbers["logits_mean"]
            ),
            "logits_std_population_delta_bf16_minus_fp32": float_difference(
                fp32_numbers["logits_std_population"], bf16_numbers["logits_std_population"]
            ),
            "both_losses_finite": bool(fp32_numbers["loss_is_finite"]) and bool(bf16_numbers["loss_is_finite"]),
            "both_logits_finite": bool(fp32_numbers["logits_all_finite"]) and bool(bf16_numbers["logits_all_finite"]),
        },
    }


def public_runtime_metadata(runtime: RuntimeInfo) -> dict[str, JSONValue]:
    """Keep reproducibility facts while deliberately avoiding machine-identifying metadata."""

    software: dict[str, JSONValue] = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.version() is not None else None,
    }
    hardware: dict[str, JSONValue] = {"device_type": runtime.device.type}
    if runtime.device.type == "cuda":
        hardware.update(
            {
                "device_name": torch.cuda.get_device_name(runtime.device),
                "device_index": runtime.device.index if runtime.device.index is not None else torch.cuda.current_device(),
                "total_memory_bytes": int(torch.cuda.get_device_properties(runtime.device).total_memory),
            }
        )
    return {
        "execution_label": runtime.execution_label,
        "is_cuda_experiment": runtime.is_cuda_experiment,
        "hardware": hardware,
        "software": software,
        "privacy": {
            "omitted_fields": ["hostname", "username", "ip_address", "device_uuid", "local_paths", "environment_variables"],
        },
        "warnings": list(runtime.warnings),
    }


def safe_command_metadata(config: ExperimentConfig, runtime: RuntimeInfo, accumulation_only: bool) -> dict[str, JSONValue]:
    """Record all experimental knobs without serializing a local command path."""

    return {
        "entrypoint": "profiling/mixed_precision.py",
        "output_path": "omitted_to_avoid_local_path_disclosure",
        "accumulation_only": accumulation_only,
        "device": str(runtime.device),
        "cpu_test_mode": not runtime.is_cuda_experiment,
        "model_size": config.model_size,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "vocab_size": config.vocab_size,
        "warmup_steps": config.warmup_steps,
        "measurement_steps": config.measurement_steps,
        "mode": config.mode,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
    }


def write_json(output_path: Path, payload: dict[str, JSONValue]) -> None:
    """Atomically write a small strict-JSON result file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
    ) as temporary_file:
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        temporary_file.write("\n")
        temporary_path = Path(temporary_file.name)
    temporary_path.replace(output_path)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数。")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the four mixed-precision accumulation cases, a ToyModel dtype probe, and FP32/BF16 LM benchmarks."
    )
    parser.add_argument("--output", type=Path, default=Path("results/mixed_precision.json"), help="轻量 JSON 结果文件。")
    parser.add_argument("--device", default="cuda", help="默认 cuda；仅与 --cpu-test-mode 联用时可设为 cpu。")
    parser.add_argument(
        "--cpu-test-mode",
        action="store_true",
        help="允许 CUDA 不可用时执行小型 CPU 正确性冒烟；结果不会被标记为 CUDA 实验。",
    )
    parser.add_argument("--accumulation-only", action="store_true", help="只运行不依赖 CUDA 的四种累加实验。")
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_CONFIGS),
        default=None,
        help="CUDA 默认 small；CPU 测试默认 smoke。smoke 仅用于冒烟，不是题面语言模型规模。",
    )
    parser.add_argument("--batch-size", type=positive_int, default=None, help="CUDA 默认 4；CPU 测试默认 1。")
    parser.add_argument("--context-length", type=positive_int, default=None, help="CUDA 默认 512；CPU 测试默认 32。")
    parser.add_argument("--vocab-size", type=positive_int, default=None, help="CUDA 默认 10000；CPU 测试默认 128。")
    parser.add_argument("--warmup", type=nonnegative_int, default=None, help="CUDA 默认 5；CPU 测试默认 1。")
    parser.add_argument("--steps", type=positive_int, default=None, help="CUDA 默认 10；CPU 测试默认 1。")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step", "all"),
        default="train_step",
        help="被测工作负载；all 会依次测三种模式。",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="模型初始化和随机 batch 的种子。")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="train_step 的 AdamW 学习率。")
    return parser


def resolve_experiment_config(args: argparse.Namespace, runtime: RuntimeInfo) -> ExperimentConfig:
    if args.learning_rate <= 0.0 or not math.isfinite(args.learning_rate):
        raise ValueError("--learning-rate 必须是有限正数。")
    cpu_defaults = not runtime.is_cuda_experiment
    return ExperimentConfig(
        model_size=args.model_size or ("smoke" if cpu_defaults else "small"),
        batch_size=args.batch_size if args.batch_size is not None else (1 if cpu_defaults else 4),
        context_length=args.context_length if args.context_length is not None else (32 if cpu_defaults else 512),
        vocab_size=args.vocab_size if args.vocab_size is not None else (128 if cpu_defaults else 10_000),
        warmup_steps=args.warmup if args.warmup is not None else (1 if cpu_defaults else 5),
        measurement_steps=args.steps if args.steps is not None else (1 if cpu_defaults else 10),
        mode=args.mode,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )


def selected_modes(mode: str) -> tuple[BenchmarkMode, ...]:
    if mode == "all":
        return ("forward", "forward_backward", "train_step")
    return (cast(BenchmarkMode, mode),)


def run_experiment(args: argparse.Namespace) -> dict[str, JSONValue]:
    """Run all requested portions and return a JSON-serializable result document."""

    accumulation_results = run_accumulation_experiments()
    if args.accumulation_only:
        return {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed_accumulation_only",
            "accumulation": {"cases": accumulation_results},
            "metadata": {
                "entrypoint": "profiling/mixed_precision.py",
                "output_path": "omitted_to_avoid_local_path_disclosure",
                "privacy": {"omitted_fields": ["hostname", "username", "ip_address", "local_paths"]},
                "software": {"python_version": platform.python_version(), "pytorch_version": torch.__version__},
            },
        }

    runtime = resolve_runtime(args.device, args.cpu_test_mode)
    config = resolve_experiment_config(args, runtime)
    model_config = MODEL_CONFIGS[config.model_size]
    input_ids, targets = make_random_batch(
        config.batch_size, config.context_length, config.vocab_size, runtime.device, config.seed + 1
    )
    toy_probe = run_toymodel_dtype_probe(runtime.device, config.seed)
    mode_results: dict[str, JSONValue] = {}
    for mode in selected_modes(config.mode):
        fp32_result = run_precision_benchmark(
            "fp32", mode, model_config, config, runtime, input_ids, targets
        )
        bf16_result = run_precision_benchmark(
            "bf16", mode, model_config, config, runtime, input_ids, targets
        )
        mode_results[mode] = {
            "fp32": fp32_result,
            "bf16_autocast": bf16_result,
            "comparison": make_precision_comparison(fp32_result, bf16_result),
        }

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if runtime.is_cuda_experiment else "completed_cpu_smoke_test_not_comparable_to_cuda",
        "accumulation": {"cases": accumulation_results},
        "toy_model_dtype_probe": toy_probe,
        "language_model_benchmark": {
            "model_config": asdict(model_config),
            "experiment_config": asdict(config),
            "modes": mode_results,
        },
        "metadata": {
            "runtime": public_runtime_metadata(runtime),
            "command": safe_command_metadata(config, runtime, accumulation_only=False),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_experiment(args)
        write_json(args.output, result)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote mixed-precision results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
