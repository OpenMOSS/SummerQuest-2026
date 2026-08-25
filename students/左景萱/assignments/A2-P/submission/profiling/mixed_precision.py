#!/usr/bin/env python3
"""Reproducible mixed-precision experiments for SummerQuest CS336 A2-P.

The CUDA path is the authoritative experiment.  ``--device cpu`` deliberately
uses a tiny configuration so contributors can validate the control flow without
claiming GPU timing, CUDA autocast, or CUDA allocator evidence.

The emitted JSON is suitable for the public submission: it contains logical
commands and public hardware/software labels, but never records argv, cwd,
hostnames, usernames, environment variables, device UUIDs, or absolute paths.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


SCHEMA_VERSION = "cs336.a2p.mixed-precision.v1"
PUBLIC_RESULT_PATH = Path("results/mixed_precision.json")

OFFICIAL_MODEL_DIMENSIONS = {
    "small": {"d_model": 768, "d_ff": 3_072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1_024, "d_ff": 4_096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1_280, "d_ff": 5_120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2_560, "d_ff": 10_240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4_608, "d_ff": 12_288, "num_layers": 50, "num_heads": 36},
}

CPU_DRY_RUN_MODEL = {
    "model_size": "cpu_micro_dry_run",
    "d_model": 32,
    "d_ff": 64,
    "num_layers": 2,
    "num_heads": 4,
    "vocab_size": 64,
    "batch_size": 1,
    "context_length": 8,
    "warmup_steps": 1,
    "measurement_steps": 2,
}


def official_model_config(model_size: str) -> dict[str, int | str]:
    try:
        dimensions = OFFICIAL_MODEL_DIMENSIONS[model_size]
    except KeyError as exc:
        raise ValueError(f"unknown model size: {model_size}") from exc
    return {
        "model_size": model_size,
        **dimensions,
        "vocab_size": 10_000,
        "batch_size": 4,
        "context_length": 512,
        "warmup_steps": 5,
        "measurement_steps": 10,
    }


class EvidenceError(RuntimeError):
    """Raised when a run cannot support its advertised evidence contract."""


class ToyModel(nn.Module):
    """The exact ToyModel architecture from the pinned Assignment 2 PDF."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bf16_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def precision_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "bf16":
        raise ValueError(f"unsupported precision: {precision}")
    return bf16_autocast(device)


def run_accumulation_experiment() -> dict[str, Any]:
    """Execute the four accumulation snippets from the pinned PDF verbatim."""

    # Snippet 1: FP32 value, FP32 accumulator.
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    fp32_fp32 = s

    # Snippet 2: FP16 value, FP16 accumulator.
    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    fp16_fp16 = s

    # Snippet 3: FP16 value, FP32 accumulator, implicit in-place cast.
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    fp16_fp32_implicit = s

    # Snippet 4: FP16 value, explicitly cast to the FP32 accumulator dtype.
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    fp16_fp32_explicit = s

    values = (
        ("fp32_value_fp32_accumulator", "float32", "float32", fp32_fp32),
        ("fp16_value_fp16_accumulator", "float16", "float16", fp16_fp16),
        ("fp16_value_fp32_accumulator_implicit_cast", "float16", "float32", fp16_fp32_implicit),
        ("fp16_value_fp32_accumulator_explicit_cast", "float16", "float32", fp16_fp32_explicit),
    )
    records = [
        {
            "case": case,
            "input_dtype": input_dtype,
            "accumulator_dtype": accumulator_dtype,
            "iterations": 1_000,
            "increment": 0.01,
            "actual_value": float(value.item()),
            "mathematical_value": 10.0,
            "absolute_error": abs(float(value.item()) - 10.0),
        }
        for case, input_dtype, accumulator_dtype, value in values
    ]
    return {
        "source": "pinned_assignment_pdf_mixed_precision_accumulation",
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "executed_as_written": True,
        "records": records,
    }


def run_toy_model_dtype_probe(device: torch.device, seed: int) -> dict[str, Any]:
    """Record actual ToyModel dtypes under BF16 autocast."""

    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise EvidenceError("the selected CUDA device does not support BF16")

    seed_everything(seed)
    model = ToyModel(in_features=16, out_features=7).to(device=device, dtype=torch.float32)
    model.train()
    inputs = torch.randn(8, 16, device=device, dtype=torch.float32)
    targets = torch.randint(0, 7, (8,), device=device)
    observed: dict[str, str] = {}

    def record_output(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name] = dtype_name(output.dtype)

        return hook

    handles = [
        model.fc1.register_forward_hook(record_output("fc1_output")),
        model.ln.register_forward_hook(record_output("layer_norm_output")),
    ]
    try:
        model.zero_grad(set_to_none=True)
        with bf16_autocast(device):
            parameter_dtypes_inside_autocast = {name: dtype_name(parameter.dtype) for name, parameter in model.named_parameters()}
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        synchronize(device)
        loss.backward()
        synchronize(device)
        gradient_dtypes = {name: dtype_name(parameter.grad.dtype) for name, parameter in model.named_parameters() if parameter.grad is not None}
        result = {
            "status": "ok",
            "authoritative_cuda_bf16": device.type == "cuda",
            "device_type": device.type,
            "autocast_enabled": True,
            "autocast_dtype": "bfloat16",
            "parameter_dtypes": parameter_dtypes_inside_autocast,
            "parameter_dtype_set": sorted(set(parameter_dtypes_inside_autocast.values())),
            "fc1_output_dtype": observed.get("fc1_output"),
            "layer_norm_output_dtype": observed.get("layer_norm_output"),
            "logits_dtype": dtype_name(logits.dtype),
            "loss_dtype": dtype_name(loss.dtype),
            "loss_value": float(loss.detach().float().item()),
            "gradient_dtypes": gradient_dtypes,
            "gradient_dtype_set": sorted(set(gradient_dtypes.values())),
        }
    finally:
        for handle in handles:
            handle.remove()

    required = ("fc1_output_dtype", "layer_norm_output_dtype", "logits_dtype", "loss_dtype")
    if any(result[field] is None for field in required):
        raise EvidenceError("ToyModel dtype hooks did not capture every required output")
    if not result["parameter_dtypes"] or not result["gradient_dtypes"]:
        raise EvidenceError("ToyModel parameter or gradient dtype evidence is empty")
    if device.type == "cuda" and result["autocast_dtype"] != "bfloat16":
        raise EvidenceError("authoritative ToyModel evidence must use CUDA BF16 autocast")
    return result


def summarize_timings(samples_ms: list[float]) -> dict[str, float | int]:
    if len(samples_ms) < 2 or any(not math.isfinite(value) or value <= 0 for value in samples_ms):
        raise EvidenceError("timing evidence requires at least two positive finite samples")
    mean_ms = statistics.fmean(samples_ms)
    sample_std_ms = statistics.stdev(samples_ms)
    return {
        "sample_count": len(samples_ms),
        "mean_ms": mean_ms,
        "sample_std_ms": sample_std_ms,
        "cv": sample_std_ms / mean_ms,
        "cv_percent": 100.0 * sample_std_ms / mean_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _fixed_batch(config: dict[str, int | str], seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    shape = (int(config["batch_size"]), int(config["context_length"]))
    vocab_size = int(config["vocab_size"])
    input_ids = torch.randint(0, vocab_size, shape, generator=generator, device="cpu")
    targets = torch.randint(0, vocab_size, shape, generator=generator, device="cpu")
    return input_ids.to(device), targets.to(device)


def _cuda_peak_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "peak_active_bytes": None,
        }
    stats = torch.cuda.memory_stats(device)
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_active_bytes": int(stats["active_bytes.all.peak"]),
    }


def _forward_backward_step(
    model: nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    precision: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Run and separately time the two passes required by the pinned PDF.

    Gradient clearing is intentionally outside both timing boundaries.  There
    is no optimizer step: the upstream ``benchmarking_mixed_precision`` task
    asks for forward and backward timings for each model size.
    """

    model.zero_grad(set_to_none=True)
    synchronize(device)
    forward_started = time.perf_counter()
    with precision_context(device, precision):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    synchronize(device)
    forward_ms = (time.perf_counter() - forward_started) * 1_000.0

    synchronize(device)
    backward_started = time.perf_counter()
    loss.backward()
    synchronize(device)
    backward_ms = (time.perf_counter() - backward_started) * 1_000.0
    return loss, logits, forward_ms, backward_ms


def run_language_model_precision(
    *,
    precision: str,
    device: torch.device,
    config: dict[str, int | str],
    seed: int,
) -> dict[str, Any]:
    """Run one fixed full-train-step precision case."""

    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"unsupported precision: {precision}")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise EvidenceError("the selected CUDA device does not support BF16")

    seed_everything(seed)
    model = BasicsTransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
    ).to(device=device, dtype=torch.float32)
    model.train()
    input_ids, targets = _fixed_batch(config, seed, device)

    warmup_steps = int(config["warmup_steps"])
    measurement_steps = int(config["measurement_steps"])
    for _ in range(warmup_steps):
        _forward_backward_step(model, input_ids, targets, precision=precision, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_timings_ms: list[float] = []
    raw_forward_ms: list[float] = []
    raw_backward_ms: list[float] = []
    losses: list[float] = []
    final_logits: torch.Tensor | None = None
    for _ in range(measurement_steps):
        loss, logits, forward_ms, backward_ms = _forward_backward_step(
            model,
            input_ids,
            targets,
            precision=precision,
            device=device,
        )
        raw_forward_ms.append(forward_ms)
        raw_backward_ms.append(backward_ms)
        raw_timings_ms.append(forward_ms + backward_ms)
        losses.append(float(loss.detach().float().item()))
        final_logits = logits.detach()

    if final_logits is None:
        raise EvidenceError("language-model benchmark produced no measurement step")
    if len(raw_timings_ms) != measurement_steps or len(losses) != measurement_steps:
        raise EvidenceError("language-model benchmark did not produce the requested number of samples")
    if any(not math.isfinite(value) for value in losses):
        raise EvidenceError("language-model benchmark produced a non-finite loss")

    timing = summarize_timings(raw_timings_ms)
    forward_timing = summarize_timings(raw_forward_ms)
    backward_timing = summarize_timings(raw_backward_ms)
    memory = _cuda_peak_memory(device)
    record = {
        "status": "ok",
        "precision": precision,
        "autocast": precision == "bf16",
        "autocast_dtype": "bfloat16" if precision == "bf16" else None,
        "model_parameter_dtype_set": sorted({dtype_name(parameter.dtype) for parameter in model.parameters()}),
        "configuration": dict(config),
        "measurement_mode": "forward_backward",
        "optimizer_step_included": False,
        "timing": {
            "clock": "time.perf_counter",
            "unit": "milliseconds",
            "cuda_synchronize_before_and_after_each_measurement": device.type == "cuda",
            "data_generation_and_initialization_timed": False,
            "gradient_clearing_timed": False,
            "raw_ms": raw_timings_ms,
            **timing,
        },
        "phase_timings": {
            "forward_including_loss": {
                "raw_ms": raw_forward_ms,
                **forward_timing,
            },
            "backward": {
                "raw_ms": raw_backward_ms,
                **backward_timing,
            },
        },
        "memory": {
            "source": "torch.cuda.memory_stats" if device.type == "cuda" else None,
            "authoritative": device.type == "cuda",
            **memory,
        },
        "loss": {
            "raw": losses,
            "first": losses[0],
            "last": losses[-1],
            "change": losses[-1] - losses[0],
            "all_finite": True,
        },
        "final_logits": {
            "dtype": dtype_name(final_logits.dtype),
            "mean": float(final_logits.float().mean().item()),
            "std": float(final_logits.float().std().item()),
            "l2_norm": float(torch.linalg.vector_norm(final_logits.float()).item()),
            "all_finite": bool(torch.isfinite(final_logits).all().item()),
        },
    }
    if record["final_logits"]["all_finite"] is not True:
        raise EvidenceError("language-model benchmark produced non-finite logits")
    return record


def run_language_model_comparison(device: torch.device, seed: int, model_size: str = "small") -> dict[str, Any]:
    config = official_model_config(model_size) if device.type == "cuda" else dict(CPU_DRY_RUN_MODEL)
    records: dict[str, dict[str, Any]] = {}
    for precision in ("fp32", "bf16"):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            synchronize(device)
        try:
            records[precision] = run_language_model_precision(
                precision=precision,
                device=device,
                config=config,
                seed=seed,
            )
        except Exception as error:
            if not _is_oom(error):
                raise
            records[precision] = {
                "status": "oom",
                "precision": precision,
                "configuration": dict(config),
                "measurement_mode": "forward_backward",
                "optimizer_step_included": False,
                "error_type": type(error).__name__,
                "error": "CUDA out of memory; configuration and failure type retained without private diagnostics",
                "memory": {
                    "source": "torch.cuda.memory_stats" if device.type == "cuda" else None,
                    "authoritative": device.type == "cuda",
                    **_cuda_peak_memory(device),
                },
            }
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                synchronize(device)

    fp32 = records["fp32"]
    bf16 = records["bf16"]
    comparison: dict[str, Any] | None = None
    if fp32["status"] == bf16["status"] == "ok":
        fp32_mean = float(fp32["timing"]["mean_ms"])
        bf16_mean = float(bf16["timing"]["mean_ms"])
        comparison = {
            "bf16_speedup_over_fp32": fp32_mean / bf16_mean,
            "bf16_forward_speedup_over_fp32": float(fp32["phase_timings"]["forward_including_loss"]["mean_ms"]) / float(bf16["phase_timings"]["forward_including_loss"]["mean_ms"]),
            "bf16_backward_speedup_over_fp32": float(fp32["phase_timings"]["backward"]["mean_ms"]) / float(bf16["phase_timings"]["backward"]["mean_ms"]),
            "first_loss_absolute_difference": abs(float(fp32["loss"]["first"]) - float(bf16["loss"]["first"])),
            "final_loss_absolute_difference": abs(float(fp32["loss"]["last"]) - float(bf16["loss"]["last"])),
            "final_logits_mean_absolute_difference": abs(float(fp32["final_logits"]["mean"]) - float(bf16["final_logits"]["mean"])),
            "peak_allocated_bytes_difference": (int(bf16["memory"]["peak_allocated_bytes"]) - int(fp32["memory"]["peak_allocated_bytes"]) if device.type == "cuda" else None),
            "peak_reserved_bytes_difference": (int(bf16["memory"]["peak_reserved_bytes"]) - int(fp32["memory"]["peak_reserved_bytes"]) if device.type == "cuda" else None),
            "peak_active_bytes_difference": (int(bf16["memory"]["peak_active_bytes"]) - int(fp32["memory"]["peak_active_bytes"]) if device.type == "cuda" else None),
        }
    return {
        "authoritative_cuda_benchmark": device.type == "cuda",
        "same_configuration_and_seed_for_both_precisions": True,
        "measurement_boundary": ["forward", "loss", "backward"],
        "gradient_clearing_outside_timing": True,
        "optimizer_step_included": False,
        "requested_cuda_model_size": model_size,
        "configuration": config,
        "records": records,
        "comparison": comparison,
    }


def _is_oom(error: BaseException) -> bool:
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", torch.OutOfMemoryError)
    return isinstance(error, (torch.OutOfMemoryError, cuda_oom)) or "out of memory" in str(error).lower()


def public_environment(device: torch.device) -> dict[str, str | None]:
    return {
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "pytorch_version": str(torch.__version__),
        "pytorch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
    }


def logical_command(
    device: torch.device,
    seed: int,
    model_size: str = "small",
    output: Path = PUBLIC_RESULT_PATH,
) -> list[str]:
    return [
        "python",
        "profiling/mixed_precision.py",
        "--device",
        device.type,
        "--model-size",
        model_size,
        "--seed",
        str(seed),
        "--output",
        output.as_posix(),
    ]


def build_payload(
    device: torch.device,
    seed: int,
    model_size: str = "small",
    output: Path = PUBLIC_RESULT_PATH,
) -> dict[str, Any]:
    authoritative = device.type == "cuda"
    language_model_benchmark = run_language_model_comparison(device, seed, model_size)
    precision_statuses = {precision: record.get("status") for precision, record in language_model_benchmark["records"].items()}
    status = "oom" if "oom" in precision_statuses.values() else "ok"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "authoritative": authoritative,
        "diagnostic_only": not authoritative,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "assignment": {
            "name": "A2-P",
            "handout_version": "26.1.4-rc.3",
            "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        },
        "logical_command": logical_command(device, seed, model_size, output),
        "logical_result_path": output.as_posix(),
        "environment": public_environment(device),
        "seed": seed,
        "accumulation": run_accumulation_experiment(),
        "toy_model": run_toy_model_dtype_probe(device, seed),
        "language_model_benchmark": language_model_benchmark,
        "limitations": (
            (
                [
                    "At least one requested precision exhausted CUDA memory; the attempted configuration and precision-level status are retained.",
                ]
                if status == "oom"
                else []
            )
            if authoritative
            else [
                "CPU mode is a micro dry-run and is not valid CUDA BF16 autocast evidence.",
                "CPU timing and null CUDA memory fields must not be used in the A2-P report.",
            ]
        ),
        "privacy": {
            "logical_paths_only": True,
            "hostname_recorded": False,
            "username_recorded": False,
            "absolute_paths_recorded": False,
            "device_uuid_recorded": False,
        },
    }
    # Reject accidental non-standard NaN/Infinity values before publication.
    json.dumps(payload, allow_nan=False)
    return payload


def safe_relative_output(path: Path) -> Path:
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("--output must be a normalized relative path without parent traversal")
    if path.suffix.lower() != ".json":
        raise ValueError("--output must use a .json suffix")
    if path.parts[0] != "results":
        raise ValueError("--output must stay below the public results/ directory")
    if path.is_symlink():
        raise ValueError("--output must not be a symbolic link")
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed A2-P mixed-precision experiments.")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="CUDA runs the authoritative matrix; CPU runs a non-authoritative micro dry-run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-size",
        choices=tuple(OFFICIAL_MODEL_DIMENSIONS),
        default="small",
        help="one pinned PDF model size per CUDA invocation; CPU always uses the micro dry-run",
    )
    parser.add_argument("--output", type=Path, default=PUBLIC_RESULT_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output: Path | None = None
    device = torch.device(args.device)
    try:
        output = safe_relative_output(args.output)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise EvidenceError("CUDA was requested but is unavailable")
        payload = build_payload(device, args.seed, args.model_size, output)
    except Exception as exc:  # noqa: BLE001 - emit a sanitized fail-closed result
        payload = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "authoritative": False,
            "diagnostic_only": True,
            "requested_device_type": device.type,
            "requested_cuda_model_size": args.model_size,
            "logical_command": logical_command(device, args.seed, args.model_size, output or PUBLIC_RESULT_PATH),
            "error_type": type(exc).__name__,
            "error": "mixed-precision experiment failed; inspect private local diagnostics",
        }
        if output is not None:
            atomic_write_json(output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 2

    atomic_write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
