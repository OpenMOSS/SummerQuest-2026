from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from profiling.benchmark import BenchmarkConfig, run_benchmark
from profiling.config import (
    autocast_context,
    cuda_memory_metrics,
    public_command,
    public_environment,
    resolve_device,
    seed_everything,
    synchronize,
)


class ToyModel(nn.Module):
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


def accumulation_experiment() -> dict[str, Any]:
    values: dict[str, float] = {}
    accumulator = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float32)
    values["fp32_accumulator_fp32_input"] = float(accumulator)

    accumulator = torch.tensor(0.0, dtype=torch.float16)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float16)
    values["fp16_accumulator_fp16_input"] = float(accumulator)

    accumulator = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float16)
    values["fp32_accumulator_fp16_input"] = float(accumulator)

    accumulator = torch.tensor(0.0, dtype=torch.float32)
    for _ in range(1000):
        value = torch.tensor(0.01, dtype=torch.float16)
        accumulator += value.type(torch.float32)
    values["fp32_accumulator_explicit_cast_input"] = float(accumulator)
    return {
        "iterations": 1000,
        "increment": 0.01,
        "expected": 10.0,
        "outputs": values,
    }


def _dtype_hook(storage: dict[str, str], name: str):
    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if isinstance(output, tuple):
            output = output[0]
        storage[name] = str(output.dtype).replace("torch.", "")

    return hook


def run_toy_case(
    *,
    dtype: str,
    device: torch.device,
    state_dict: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    warmup: int,
    steps: int,
) -> dict[str, Any]:
    model = ToyModel(inputs.shape[-1], int(targets.max()) + 1).to(device)
    model.load_state_dict(state_dict)
    model.train()
    observed: dict[str, str] = {"parameter": str(next(model.parameters()).dtype).replace("torch.", "")}
    hooks = [
        model.fc1.register_forward_hook(_dtype_hook(observed, "fc1_output")),
        model.ln.register_forward_hook(_dtype_hook(observed, "layer_norm_output")),
        model.fc2.register_forward_hook(_dtype_hook(observed, "logits")),
    ]
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            logits = model(inputs)
            loss = loss_fn(logits, targets)
        loss.backward()
        synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_timings_ms: list[float] = []
    losses: list[float] = []
    first_logits: torch.Tensor | None = None
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        with autocast_context(device, dtype):
            logits = model(inputs)
            loss = loss_fn(logits, targets)
        loss.backward()
        synchronize(device)
        raw_timings_ms.append((time.perf_counter() - started) * 1000)
        losses.append(float(loss.detach()))
        if first_logits is None:
            first_logits = logits.detach().float().cpu()
    observed["loss"] = str(loss.dtype).replace("torch.", "")
    observed["gradient"] = str(next(parameter.grad for parameter in model.parameters() if parameter.grad is not None).dtype).replace("torch.", "")
    for hook in hooks:
        hook.remove()
    mean = sum(raw_timings_ms) / len(raw_timings_ms)
    variance = (
        sum((value - mean) ** 2 for value in raw_timings_ms) / (len(raw_timings_ms) - 1)
        if len(raw_timings_ms) > 1
        else 0.0
    )
    return {
        "dtype": dtype,
        "observed_dtypes": observed,
        "raw_timings_ms": raw_timings_ms,
        "mean_ms": mean,
        "sample_std_ms": math.sqrt(variance),
        "cv": math.sqrt(variance) / mean if mean else 0.0,
        "losses": losses,
        "first_logits": first_logits,
        "memory": cuda_memory_metrics(device),
    }


def run_mixed_precision(
    *,
    device_name: str,
    output_path: str | Path,
    warmup: int,
    steps: int,
    seed: int,
    model_size: str,
    batch_size: int,
    context_length: int,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    seed_everything(seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    toy_device = device
    seed_everything(seed)
    toy_model = ToyModel(32, 7)
    state_dict = {key: value.detach().clone() for key, value in toy_model.state_dict().items()}
    toy_inputs = torch.randn(batch_size, 32, device=toy_device)
    toy_targets = torch.randint(0, 7, (batch_size,), device=toy_device)
    toy_results = {}
    for dtype in ("fp32", "bf16"):
        seed_everything(seed)
        toy_results[dtype] = run_toy_case(
            dtype=dtype,
            device=toy_device,
            state_dict=state_dict,
            inputs=toy_inputs,
            targets=toy_targets,
            warmup=warmup,
            steps=steps,
        )
    fp32_logits = toy_results["fp32"].pop("first_logits")
    bf16_logits = toy_results["bf16"].pop("first_logits")
    numeric_comparison = {
        "max_abs_logit_difference": float((fp32_logits - bf16_logits).abs().max()),
        "mean_abs_logit_difference": float((fp32_logits - bf16_logits).abs().mean()),
    }

    command = public_command(
        "profiling.mixed_precision",
        [
            "--device",
            device_name,
            "--output",
            str(output_path),
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--seed",
            str(seed),
            "--model-size",
            model_size,
            "--batch-size",
            str(batch_size),
            "--context-length",
            str(context_length),
        ],
    )
    language_model = {}
    for dtype in ("fp32", "bf16"):
        seed_everything(seed)
        language_model[dtype] = run_benchmark(
            BenchmarkConfig(
                model_size=model_size,
                batch_size=batch_size,
                context_length=context_length,
                mode="forward_backward",
                warmup=warmup,
                steps=steps,
                dtype=dtype,
                device=device_name,
                seed=seed,
            ),
            command=command,
        )
    result = {
        "status": "success",
        "experiment": "mixed_precision",
        "command": command,
        "environment": public_environment(device),
        "accumulation": accumulation_experiment(),
        "toy_model": {
            "architecture": "Linear(32, 10) -> LayerNorm(10) -> Linear(10, 7)",
            "batch_size": batch_size,
            "warmup": warmup,
            "steps": steps,
            "cases": toy_results,
            "numeric_comparison": numeric_comparison,
        },
        "language_model": language_model,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P mixed precision experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    result = run_mixed_precision(
        device_name=args.device,
        output_path=args.output,
        warmup=args.warmup,
        steps=args.steps,
        seed=args.seed,
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
