"""Accumulation, ToyModel dtype, and FP32/BF16 language-model comparison."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional

from cs336_basics.model import BasicsTransformerLM
from profiling.common import (
    MIB,
    MODEL_CONFIGS,
    configure_gpu,
    gpu_metadata,
    read_json,
    summarize_timings,
    write_json,
)

BENCHMARK_MODEL_SIZES = ("small", "medium", "large", "xl", "10b")


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.fc1(inputs))
        hidden = self.ln(hidden)
        return self.fc2(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--single-model-size",
        choices=BENCHMARK_MODEL_SIZES,
    )
    parser.add_argument("--single-dtype", choices=("fp32", "bf16"))
    return parser.parse_args()


def accumulation_cases() -> list[dict]:
    results: list[dict] = []

    accumulator = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float32)
    results.append(
        {
            "case": "fp32_input_fp32_accumulator",
            "result": accumulator.item(),
        }
    )

    accumulator = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float16)
    results.append(
        {
            "case": "fp16_input_fp16_accumulator",
            "result": accumulator.item(),
        }
    )

    accumulator = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float16)
    results.append(
        {
            "case": "fp16_input_fp32_accumulator",
            "result": accumulator.item(),
        }
    )

    accumulator = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        value = torch.tensor(0.01, dtype=torch.float16)
        accumulator += value.to(torch.float32)
    results.append(
        {
            "case": "fp16_then_cast_fp32_accumulator",
            "result": accumulator.item(),
        }
    )
    for result in results:
        result["absolute_error_from_10"] = abs(result["result"] - 10.0)
    return results


def inspect_toy_model() -> dict:
    torch.manual_seed(2026)
    model = ToyModel(16, 7).cuda()
    inputs = torch.randn(32, 16, device="cuda")
    targets = torch.randint(0, 7, (32,), device="cuda")
    observed: dict[str, str] = {}
    hooks = [
        model.fc1.register_forward_hook(lambda _module, _inputs, output: observed.update(fc1_output=str(output.dtype))),
        model.ln.register_forward_hook(lambda _module, _inputs, output: observed.update(layernorm_output=str(output.dtype))),
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = functional.cross_entropy(logits, targets)
    loss.backward()
    for hook in hooks:
        hook.remove()
    observed.update(
        {
            "parameters": str(next(model.parameters()).dtype),
            "logits": str(logits.dtype),
            "loss": str(loss.dtype),
            "gradients": str(next(model.parameters()).grad.dtype),
            "loss_value": loss.item(),
            "logits_mean": logits.float().mean().item(),
        }
    )
    return observed


def benchmark_language_model(model_size: str, dtype: str) -> dict:
    torch.manual_seed(2026)
    row = {
        "model_size": model_size,
        "dtype": dtype,
        "batch_size": 4,
        "context_length": 512,
        "mode": "forward_backward",
        "warmup_steps": 5,
        "measurement_steps": 10,
        "raw_timings_ms": [],
        "mean_ms": "",
        "std_ms": "",
        "cv": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "loss": "",
        "logits_mean": "",
        "parameter_count": "",
        "fp32_parameter_storage_mib": "",
        "capacity_limit_mib": "",
        "capacity_precheck": "",
        "status": "error",
        "error_type": "",
        "failure_stage": "setup",
    }

    def context():
        if dtype == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    try:
        if model_size == "10b":
            with torch.device("meta"):
                meta_model = BasicsTransformerLM(
                    vocab_size=10_000,
                    context_length=512,
                    **MODEL_CONFIGS[model_size],
                )
            parameter_count = sum(parameter.numel() for parameter in meta_model.parameters())
            parameter_storage_mib = sum(
                parameter.numel() * parameter.element_size()
                for parameter in meta_model.parameters()
            ) / MIB
            row.update(
                {
                    "parameter_count": parameter_count,
                    "fp32_parameter_storage_mib": parameter_storage_mib,
                    "capacity_limit_mib": 23 * 1024,
                }
            )
            if parameter_storage_mib > 23 * 1024:
                row.update(
                    {
                        "status": "capacity_exceeded",
                        "error_type": "AllocatorCapacityPrecheck",
                        "failure_stage": "setup/parameters",
                        "capacity_precheck": (
                            "FP32 parameters alone exceed the allocator limit; "
                            "BF16 autocast does not downcast stored parameters"
                        ),
                    }
                )
                return row

        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=512,
            **MODEL_CONFIGS[model_size],
        ).cuda()
        tokens = torch.randint(0, 10_000, (4, 512), device="cuda")
        targets = torch.randint(0, 10_000, (4, 512), device="cuda")

        def step() -> tuple[float, float]:
            model.zero_grad(set_to_none=True)
            with context():
                logits = model(tokens)
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
            loss.backward()
            return loss.item(), logits.float().mean().item()

        row["failure_stage"] = "warmup"
        for _ in range(5):
            step()
            torch.cuda.synchronize()
        samples = []
        peak_allocated = []
        peak_reserved = []
        loss_value = logits_mean = 0.0
        row["failure_stage"] = "measurement"
        for _ in range(10):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            loss_value, logits_mean = step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            peak_allocated.append(torch.cuda.max_memory_allocated() / MIB)
            peak_reserved.append(torch.cuda.max_memory_reserved() / MIB)
        row.update(
            {
                "raw_timings_ms": samples,
                **summarize_timings(samples),
                "peak_allocated_mib": max(peak_allocated),
                "peak_reserved_mib": max(peak_reserved),
                "loss": loss_value,
                "logits_mean": logits_mean,
                "status": "ok",
                "failure_stage": "",
            }
        )
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        row["error_type"] = "OutOfMemoryError"
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
        row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
    except Exception as error:
        row["error_type"] = type(error).__name__
    return row


def run_isolated(model_size: str, dtype: str, output: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "profiling.mixed_precision",
        "--output",
        str(output),
        "--single-model-size",
        model_size,
        "--single-dtype",
        dtype,
    ]
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(command, env=environment, check=True)
    return read_json(output)


def main() -> int:
    args = parse_args()
    if (args.single_model_size is None) != (args.single_dtype is None):
        raise ValueError("--single-model-size and --single-dtype must be used together")
    if args.single_model_size is not None:
        allocator = configure_gpu()
        gpu = gpu_metadata()
        torch.backends.cuda.matmul.allow_tf32 = True
        row = benchmark_language_model(
            args.single_model_size,
            args.single_dtype,
        )
        write_json(
            args.output,
            {
                "row": row,
                "allocator": allocator,
                "gpu": gpu,
            },
        )
        return 0 if row["status"] in {"ok", "oom", "capacity_exceeded"} else 1

    comparisons: list[dict] = []
    runs: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="a2p-mixed-") as temporary:
        scratch = Path(temporary)
        for model_size in BENCHMARK_MODEL_SIZES:
            for dtype in ("fp32", "bf16"):
                payload = run_isolated(
                    model_size,
                    dtype,
                    scratch / f"{model_size}-{dtype}.json",
                )
                comparisons.append(payload["row"])
                runs.append(
                    {
                        "model_size": model_size,
                        "dtype": dtype,
                        "allocator": payload["allocator"],
                        "gpu": payload["gpu"],
                    }
                )

    allocator = configure_gpu()
    gpu = gpu_metadata()
    torch.backends.cuda.matmul.allow_tf32 = True
    payload = {
        "accumulation": accumulation_cases(),
        "toy_model_bf16_autocast": inspect_toy_model(),
        "language_model_comparison": comparisons,
        "configuration": {
            "model_sizes": list(BENCHMARK_MODEL_SIZES),
            "batch_size": 4,
            "context_length": 512,
            "mode": "forward_backward",
            "warmup_steps": 5,
            "measurement_steps": 10,
            "process_isolation": "one fresh Python process per model/dtype row",
        },
        "runs": runs,
        "allocator": allocator,
        "gpu": gpu,
        "command": ("python -m profiling.mixed_precision --output results/mixed_precision.json"),
    }
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
