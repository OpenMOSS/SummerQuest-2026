"""CUDA ToyModel dtype, latency, numerical, and peak-memory experiment."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

from profiling.benchmark import environment_metadata


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def run_one(dtype_name: str, args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    model = ToyModel(args.in_features, args.out_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    inputs = torch.randn(args.batch_size, args.in_features, generator=generator, device=device)
    targets = torch.randn(args.batch_size, args.out_features, generator=generator, device=device)
    cast_dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    enabled = dtype_name != "fp32"
    observed: dict[str, str] = {}

    def record(name):
        def hook(_module, _inputs, output):
            observed[name] = str(output.dtype)

        return hook

    handles = [model.fc1.register_forward_hook(record("fc1_output")), model.ln.register_forward_hook(record("layernorm_output"))]

    def step(record_dtypes: bool = False):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=cast_dtype, enabled=enabled):
            logits = model(inputs)
            loss = torch.nn.functional.mse_loss(logits, targets)
        if record_dtypes:
            # Autocast changes eligible operations, not parameter storage.
            observed["parameters"] = sorted({str(p.dtype) for p in model.parameters()})
            observed["logits"] = str(logits.dtype)
            observed["loss"] = str(loss.dtype)
        loss.backward()
        if record_dtypes:
            observed["gradients"] = sorted({str(p.grad.dtype) for p in model.parameters() if p.grad is not None})
        optimizer.step()
        return loss.detach().float().item()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    losses = []
    for index in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses.append(step(record_dtypes=index == 0))
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    for handle in handles:
        handle.remove()
    mean = statistics.fmean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return {
        "dtype": dtype_name,
        "observed_dtypes": observed,
        "timings_ms": timings,
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean else 0.0,
        "losses": losses,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ToyModel mixed-precision benchmark requires CUDA")
    device = torch.device("cuda")
    result = {
        "experiment": "ToyModel FP32 versus CUDA BF16 autocast",
        "config": vars(args) | {"output": str(args.output)},
        "environment": environment_metadata(device),
        "command": " ".join(sys.argv),
        "runs": [run_one(dtype, args, device) for dtype in ("fp32", "bf16")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
