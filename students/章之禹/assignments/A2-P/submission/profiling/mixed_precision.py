"""Mixed-precision accumulation, dtype probes, and benchmark comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from profiling.benchmark import run_benchmark
from profiling.common import autocast_context, hardware_metadata, resolve_device, sanitized_command, write_json


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.ln(self.relu(self.fc1(x))))


def accumulation_experiment() -> dict[str, float]:
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
        accumulator += value.to(torch.float32)
    values["fp32_accumulator_explicit_cast"] = float(accumulator)
    return values


def toy_dtype_probe(device: torch.device, dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    model = ToyModel(32, 7).to(device)
    model.train()
    inputs = torch.randn(4, 32, device=device)
    captures: dict[str, torch.dtype] = {}

    def capture(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            captures[name] = output.dtype

        return hook

    model.fc1.register_forward_hook(capture("fc1_output"))
    model.ln.register_forward_hook(capture("layernorm_output"))
    model.fc2.register_forward_hook(capture("logits"))
    model.zero_grad(set_to_none=True)
    with autocast_context(device, dtype):
        output = model(inputs)
        loss = output.float().square().mean()
    loss.backward()
    gradient_dtypes = {name: parameter.grad.dtype for name, parameter in model.named_parameters() if parameter.grad is not None}
    return {
        "parameter_dtype": str(next(model.parameters()).dtype),
        "fc1_output_dtype": str(captures.get("fc1_output")),
        "layernorm_output_dtype": str(captures.get("layernorm_output")),
        "logits_dtype": str(captures.get("logits")),
        "loss_dtype": str(loss.dtype),
        "gradient_dtypes": {name: str(dtype) for name, dtype in gradient_dtypes.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-size", action="append", dest="model_sizes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "hardware": hardware_metadata(device),
        "command": sanitized_command("python profiling/mixed_precision.py", sys.argv[1:]),
        "accumulation": accumulation_experiment(),
        "toy_model": toy_dtype_probe(device),
        "benchmarks": [],
    }
    model_sizes = args.model_sizes or ["small", "medium", "large", "xl", "10b"]
    for model_size in model_sizes:
        for autocast in (False, True):
            namespace = argparse.Namespace(
                model_size=model_size,
                vocab_size=10_000,
                batch_size=args.batch_size,
                context_length=args.context_length,
                mode="forward_backward",
                warmup=args.warmup,
                steps=args.steps,
                dtype="bf16",
                autocast=autocast,
                device=args.device,
                seed=1337,
                annotate_attention=False,
                output=None,
            )
            benchmark = run_benchmark(namespace)
            benchmark["mixed_precision"] = autocast
            result["benchmarks"].append(benchmark)
            if benchmark.get("status") != "complete":
                result["status"] = "partial"
            if device.type == "cuda":
                # A failed large-model allocation can leave cached blocks
                # behind.  Each formal run is independent in the orchestrator,
                # and this is a defensive cleanup for direct multi-size use.
                torch.cuda.empty_cache()
    # A large-model OOM is useful evidence, but it must not be reported as a
    # completely successful matrix.  Keep all rows and expose the aggregate
    # status so the orchestrator and report can distinguish partial coverage.
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
