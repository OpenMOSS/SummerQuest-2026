#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.model import BasicsTransformerLM
from profiling.common import (
    MODEL_CONFIGS,
    autocast_context,
    environment_metadata,
    sample_statistics,
    set_seed,
    synchronize,
)
from profiling.io_utils import sanitized_command, utc_timestamp, write_json


class ToyModel(nn.Module):
    """The exact ToyModel architecture from the fixed assignment handout."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P mixed-precision experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    accumulation = subparsers.add_parser("accumulation", help="run the four fixed accumulation snippets")
    accumulation.add_argument("--output", type=Path, required=True)

    toy = subparsers.add_parser("toy", help="record ToyModel BF16 autocast dtypes")
    toy.add_argument("--output", type=Path, required=True)
    toy.add_argument("--device", default="cuda")
    toy.add_argument("--in-features", type=int, default=32)
    toy.add_argument("--out-features", type=int, default=8)
    toy.add_argument("--batch-size", type=int, default=4)
    toy.add_argument("--seed", type=int, default=42)

    benchmark = subparsers.add_parser("benchmark", help="compare FP32 and BF16 language-model runs")
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", default="cuda")
    benchmark.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    benchmark.add_argument("--batch-size", type=int, default=4)
    benchmark.add_argument("--context-length", type=int, default=512)
    benchmark.add_argument("--warmup", type=int, default=5)
    benchmark.add_argument("--steps", type=int, default=10)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--vocab-size", type=int, default=10_000)
    benchmark.add_argument("--d-model", type=int)
    benchmark.add_argument("--d-ff", type=int)
    benchmark.add_argument("--num-layers", type=int)
    benchmark.add_argument("--num-heads", type=int)
    return parser.parse_args()


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def update_output(path: Path, section: str, value: Any) -> None:
    payload: dict[str, Any] = {}
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"expected a JSON object in {path}")
        payload = loaded
    if section == "benchmarks":
        records = payload.setdefault(section, [])
        if not isinstance(records, list):
            raise ValueError(f"expected {section} to be a JSON list")
        records[:] = [record for record in records if record.get("run_id") != value["run_id"]]
        records.append(value)
        records.sort(key=lambda record: record["run_id"])
    else:
        payload[section] = value
    write_json(path, payload)


def accumulation_experiment() -> dict[str, Any]:
    # These four snippets intentionally mirror the fixed handout line-for-line.
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    fp32_input_fp32_accumulator = s

    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    fp16_input_fp16_accumulator = s

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    fp16_input_fp32_accumulator_implicit = s

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    fp16_input_fp32_accumulator_explicit = s

    tensors = {
        "fp32_input_fp32_accumulator": fp32_input_fp32_accumulator,
        "fp16_input_fp16_accumulator": fp16_input_fp16_accumulator,
        "fp16_input_fp32_accumulator_implicit": fp16_input_fp32_accumulator_implicit,
        "fp16_input_fp32_accumulator_explicit": fp16_input_fp32_accumulator_explicit,
    }
    return {
        "timestamp_utc": utc_timestamp(),
        "command": sanitized_command(),
        "expected_exact_sum": 10.0,
        "results": {
            name: {
                "value": tensor.item(),
                "dtype": dtype_name(tensor.dtype),
                "absolute_error": abs(tensor.item() - 10.0),
            }
            for name, tensor in tensors.items()
        },
    }


def toy_experiment(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("the formal ToyModel experiment requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("the selected CUDA device does not support BF16")
    if min(args.in_features, args.out_features, args.batch_size) < 1:
        raise SystemExit("ToyModel dimensions and batch size must be positive")

    set_seed(args.seed)
    model = ToyModel(args.in_features, args.out_features).to(device)
    x = torch.randn(args.batch_size, args.in_features, device=device)
    targets = torch.randint(args.out_features, (args.batch_size,), device=device)
    captured: dict[str, str] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = dtype_name(output.dtype)

        return hook

    handles = [
        model.fc1.register_forward_hook(capture("fc1_output")),
        model.ln.register_forward_hook(capture("layer_norm_output")),
    ]
    try:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parameter_dtypes_inside_autocast = sorted({dtype_name(parameter.dtype) for parameter in model.parameters()})
            logits = model(x)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        synchronize(device)
    finally:
        for handle in handles:
            handle.remove()

    gradient_dtypes = sorted({dtype_name(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None})
    return {
        "timestamp_utc": utc_timestamp(),
        "command": sanitized_command(),
        "configuration": {
            "autocast_dtype": "bfloat16",
            "in_features": args.in_features,
            "out_features": args.out_features,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "environment": environment_metadata(device),
        "dtypes": {
            "parameters_inside_autocast": parameter_dtypes_inside_autocast,
            "fc1_output": captured["fc1_output"],
            "layer_norm_output": captured["layer_norm_output"],
            "logits": dtype_name(logits.dtype),
            "loss": dtype_name(loss.dtype),
            "gradients": gradient_dtypes,
        },
        "numerics": {
            "loss": loss.detach().float().item(),
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "logits_mean": logits.detach().float().mean().item(),
            "logits_std": logits.detach().float().std().item(),
        },
    }


def resolved_model_config(args: argparse.Namespace) -> dict[str, int]:
    config = dict(MODEL_CONFIGS[args.model_size])
    for key in ("d_model", "d_ff", "num_layers", "num_heads"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if config["d_model"] % config["num_heads"] != 0:
        raise SystemExit("d_model must be divisible by num_heads")
    return config


def memory_metrics(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    stats = torch.cuda.memory_stats(device)
    return {
        "active_peak_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def language_model_run(
    args: argparse.Namespace,
    *,
    dtype: str,
    device: torch.device,
    config: dict[str, int],
) -> tuple[dict[str, Any], torch.Tensor]:
    set_seed(args.seed)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device)
    model.train()
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    labels = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

    def step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            logits = model(tokens)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels.reshape(-1))
        loss.backward()
        return loss.detach()

    for _ in range(args.warmup):
        step()
        synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        synchronize(device)
        start = time.perf_counter()
        loss = step()
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
        losses.append(loss.float().item())

    model.zero_grad(set_to_none=True)
    with torch.no_grad(), autocast_context(device, dtype):
        logits = model(tokens)
        final_loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), labels.reshape(-1))
    sample = logits.detach().float().reshape(-1)[:4096].cpu()
    result = {
        "dtype": dtype,
        "timings_ms": timings,
        "timing_statistics": sample_statistics(timings),
        "memory": memory_metrics(device),
        "numerics": {
            "measurement_losses": losses,
            "final_loss": final_loss.detach().float().item(),
            "logits_finite": bool(torch.isfinite(logits).all().item()),
            "logits_mean": logits.detach().float().mean().item(),
            "logits_std": logits.detach().float().std().item(),
        },
    }
    return result, sample


def benchmark_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.batch_size, args.context_length, args.steps) < 1 or args.warmup < 0:
        raise SystemExit("batch size, context length, and steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("the selected CUDA device does not support BF16")
    config = resolved_model_config(args)
    fp32, fp32_sample = language_model_run(args, dtype="fp32", device=device, config=config)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bf16, bf16_sample = language_model_run(args, dtype="bf16", device=device, config=config)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    delta = bf16_sample - fp32_sample
    denominator = torch.linalg.vector_norm(fp32_sample)
    relative_l2 = torch.linalg.vector_norm(delta) / denominator if denominator.item() != 0 else torch.tensor(0.0)
    run_id = f"{args.model_size}-bs{args.batch_size}-ctx{args.context_length}-forward_backward-fp32-vs-bf16-seed{args.seed}"
    return {
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "command": sanitized_command(),
        "configuration": {
            "model_size": args.model_size,
            **config,
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": "forward_backward",
            "warmup": args.warmup,
            "steps": args.steps,
            "seed": args.seed,
        },
        "environment": environment_metadata(device),
        "fp32": fp32,
        "bf16": bf16,
        "comparison": {
            "loss_absolute_difference": abs(bf16["numerics"]["final_loss"] - fp32["numerics"]["final_loss"]),
            "sampled_logits_relative_l2": relative_l2.item(),
            "sampled_logits_count": fp32_sample.numel(),
        },
    }


def main() -> int:
    args = parse_args()
    if args.command == "accumulation":
        update_output(args.output, "accumulation", accumulation_experiment())
    elif args.command == "toy":
        update_output(args.output, "toy_model", toy_experiment(args))
    elif args.command == "benchmark":
        update_output(args.output, "benchmarks", benchmark_experiment(args))
    else:
        raise AssertionError(f"unexpected command: {args.command}")
    print(f"updated {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
