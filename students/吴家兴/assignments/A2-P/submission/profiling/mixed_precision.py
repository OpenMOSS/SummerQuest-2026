#!/usr/bin/env python3
"""Fixed accumulation, ToyModel dtype, and FP32/BF16 benchmark experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

try:
    from .benchmark import benchmark_once
    from .common import public_environment, seed_everything, write_json
    from .config import RunConfig
except ImportError:
    from benchmark import benchmark_once
    from common import public_environment, seed_everything, write_json
    from config import RunConfig


class ToyModel(nn.Module):
    """The exact two-linear-layer model specified in the fixed upstream PDF."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def accumulation_experiment() -> dict[str, Any]:
    """Run the four snippets without algebraically replacing their loops."""

    exact = 10.0

    first = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        first += torch.tensor(0.01, dtype=torch.float32)

    second = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        second += torch.tensor(0.01, dtype=torch.float16)

    third = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        third += torch.tensor(0.01, dtype=torch.float16)

    fourth = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        value = torch.tensor(0.01, dtype=torch.float16)
        fourth += value.type(torch.float32)

    cases = [
        ("fp32_input_fp32_accumulator", first),
        ("fp16_input_fp16_accumulator", second),
        ("fp16_input_implicit_fp32_accumulator", third),
        ("fp16_input_explicit_fp32_accumulator", fourth),
    ]
    return {
        "iterations": 1000,
        "increment": 0.01,
        "exact_real_sum": exact,
        "cases": [
            {
                "name": name,
                "value": float(value.item()),
                "absolute_error": abs(float(value.item()) - exact),
                "result_dtype": _dtype_name(value.dtype),
            }
            for name, value in cases
        ],
    }


def toy_model_dtype_experiment(seed: int) -> dict[str, Any]:
    """Observe actual CUDA BF16-autocast dtypes through forward hooks."""

    seed_everything(seed)
    model = ToyModel(in_features=16, out_features=7).cuda()
    inputs = torch.randn((32, 16), device="cuda", dtype=torch.float32)
    labels = torch.randint(0, 7, (32,), device="cuda")
    activations: dict[str, str] = {}

    def capture(name: str):
        def hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            activations[name] = _dtype_name(output.dtype)

        return hook

    handles = [
        model.fc1.register_forward_hook(capture("first_linear_output")),
        model.ln.register_forward_hook(capture("layer_norm_output")),
        model.fc2.register_forward_hook(capture("logits")),
    ]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = functional.cross_entropy(logits, labels)
    loss.backward()
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()

    parameter_dtypes = sorted(
        {_dtype_name(parameter.dtype) for parameter in model.parameters()}
    )
    gradient_dtypes = sorted(
        {
            _dtype_name(parameter.grad.dtype)
            for parameter in model.parameters()
            if parameter.grad is not None
        }
    )
    return {
        "autocast": {
            "device_type": "cuda",
            "dtype": "bfloat16",
        },
        "parameter_dtypes": parameter_dtypes,
        **activations,
        "loss_dtype": _dtype_name(loss.dtype),
        "gradient_dtypes": gradient_dtypes,
        "loss_value": float(loss.detach().item()),
    }


def language_model_comparison(
    seed: int,
    warmup: int,
    steps: int,
    batch_size: int,
    context_length: int,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for dtype in ("fp32", "bf16"):
        config = RunConfig(
            model_size="small",
            batch_size=batch_size,
            context_length=context_length,
            mode="train_step",
            dtype=dtype,
            warmup_steps=warmup,
            measurement_steps=steps,
            seed=seed,
        )
        row = benchmark_once(config)
        rows[dtype] = {
            "config": config.as_dict(),
            "raw_timings_ms": json.loads(row["timings_ms"]),
            "mean_ms": row["mean_ms"],
            "sample_std_ms": row["sample_std_ms"],
            "cv": row["cv"],
            "peak_allocated_mib": row["peak_allocated_mib"],
            "peak_reserved_mib": row["peak_reserved_mib"],
            "first_loss": row["first_loss"],
            "last_loss": row["last_loss"],
        }
    return {
        "runs": rows,
        "derived": {
            "bf16_speedup_over_fp32": round(
                rows["fp32"]["mean_ms"] / rows["bf16"]["mean_ms"],
                6,
            ),
            "bf16_peak_allocated_ratio": round(
                rows["bf16"]["peak_allocated_mib"]
                / rows["fp32"]["peak_allocated_mib"],
                6,
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "schema_version": 1,
        "experiment": "mixed_precision",
        "environment": public_environment(),
        "accumulation": accumulation_experiment(),
        "toy_model_bf16_autocast": toy_model_dtype_experiment(args.seed),
        "language_model_fp32_vs_bf16": language_model_comparison(
            seed=args.seed,
            warmup=args.warmup,
            steps=args.steps,
            batch_size=args.batch_size,
            context_length=args.context_length,
        ),
    }
    write_json(args.output, payload)
    speedup = payload["language_model_fp32_vs_bf16"]["derived"][
        "bf16_speedup_over_fp32"
    ]
    print(f"mixed-precision experiments complete; BF16 speedup={speedup:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
