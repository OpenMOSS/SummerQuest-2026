from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from profiling.common import (
    RunConfig,
    build_model,
    cleanup_cuda,
    command_string,
    config_dict,
    make_batch,
    public_environment,
    require_cuda,
    run_model_step,
    safe_write_json,
    set_seed,
    synchronize,
)


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward_with_dtypes(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, str]]:
        first = self.fc1(x)
        activated = self.relu(first)
        normalized = self.ln(activated)
        logits = self.fc2(normalized)
        return logits, {
            "fc1_output": str(first.dtype),
            "layer_norm_output": str(normalized.dtype),
            "logits": str(logits.dtype),
        }


def accumulation_experiment() -> list[dict[str, float | str]]:
    outputs = []
    variants = [
        ("fp32_input_fp32_accumulator", torch.float32, torch.float32, False),
        ("fp16_input_fp16_accumulator", torch.float16, torch.float16, False),
        ("fp16_input_fp32_accumulator_implicit", torch.float16, torch.float32, False),
        ("fp16_input_fp32_accumulator_explicit", torch.float16, torch.float32, True),
    ]
    for name, input_dtype, accumulator_dtype, explicit_cast in variants:
        total = torch.tensor(0, dtype=accumulator_dtype)
        for _ in range(1000):
            value = torch.tensor(0.01, dtype=input_dtype)
            total += value.float() if explicit_cast else value
        outputs.append(
            {
                "name": name,
                "input_dtype": str(input_dtype),
                "accumulator_dtype": str(accumulator_dtype),
                "value": float(total),
                "absolute_error_from_10": abs(float(total) - 10.0),
            }
        )
    return outputs


def toy_experiment(device: torch.device, seed: int) -> dict:
    set_seed(seed)
    model = ToyModel(16, 4).to(device)
    x = torch.randn(8, 16, device=device)
    target = torch.randn(8, 4, device=device)
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, dtypes = model.forward_with_dtypes(x)
        loss = F.mse_loss(logits.float(), target)
    loss.backward()
    dtypes.update(
        {
            "parameters_inside_autocast": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "loss": str(loss.dtype),
            "gradients": sorted({str(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None}),
        }
    )
    cleanup_cuda(model, x, target, logits, loss)
    return dtypes


def time_language_model(config: RunConfig, warmup: int, steps: int) -> dict:
    device = require_cuda()
    set_seed(config.seed)
    model = build_model(config, device)
    inputs, targets = make_batch(config, device)
    try:
        for _ in range(warmup):
            run_model_step(model, inputs, targets, config, None)
            synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        raw_ms = []
        for _ in range(steps):
            model.zero_grad(set_to_none=True)
            synchronize()
            start = time.perf_counter()
            run_model_step(model, inputs, targets, config, None)
            synchronize()
            raw_ms.append((time.perf_counter() - start) * 1000)
        return {
            "status": "ok",
            **config_dict(config),
            "warmup_steps": warmup,
            "measurement_steps": steps,
            "raw_ms": raw_ms,
            "mean_ms": sum(raw_ms) / len(raw_ms),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
    finally:
        cleanup_cuda(model, inputs, targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2-P mixed-precision experiments.")
    parser.add_argument("--models", nargs="+", default=["small", "medium", "large", "xl"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda()
    payload = {
        "accumulation": accumulation_experiment(),
        "toy_model_bf16_autocast": toy_experiment(device, args.seed),
        "language_model": [],
        "command": command_string(),
        "environment": public_environment(),
    }
    for model_size in args.models:
        for dtype in ("fp32", "bf16"):
            config = RunConfig(
                model_size=model_size,
                batch_size=args.batch_size,
                context_length=args.context_length,
                mode="forward_backward",
                dtype=dtype,
                seed=args.seed,
            )
            try:
                row = time_language_model(config, args.warmup, args.steps)
            except torch.cuda.OutOfMemoryError as exc:
                cleanup_cuda()
                row = {"status": "oom", **config_dict(config), "error_type": type(exc).__name__}
            payload["language_model"].append(row)
    safe_write_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
