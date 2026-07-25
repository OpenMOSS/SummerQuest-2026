from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from profiling.config import (
    MODEL_CONFIGS,
    autocast_context,
    build_model,
    build_optimizer,
    cuda_memory_metrics,
    model_config,
    public_command,
    public_environment,
    public_path,
    resolve_device,
    sanitize_error,
    seed_everything,
    synchronize,
)
from profiling.nvtx_ranges import profile_range


MODES = ("forward", "forward_backward", "train_step")


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str = "small"
    batch_size: int = 4
    context_length: int = 512
    mode: str = "train_step"
    warmup: int = 5
    steps: int = 10
    dtype: str = "fp32"
    device: str = "cuda"
    seed: int = 0


def _validate(config: BenchmarkConfig) -> None:
    if config.model_size not in MODEL_CONFIGS:
        raise ValueError(f"unknown model size: {config.model_size}")
    if config.mode not in MODES:
        raise ValueError(f"unknown mode: {config.mode}")
    if config.batch_size < 1 or config.context_length < 1:
        raise ValueError("batch size and context length must be positive")
    if config.warmup < 0 or config.steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")


def execute_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    dtype: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if mode == "train_step":
        assert optimizer is not None
        with profile_range("optimizer/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
    elif mode == "forward_backward":
        model.zero_grad(set_to_none=True)

    grad_context = torch.no_grad() if mode == "forward" else torch.enable_grad()
    with grad_context, autocast_context(device, dtype):
        with profile_range("forward"):
            logits = model(tokens)
        loss = None
        if mode != "forward":
            with profile_range("loss"):
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    if mode != "forward":
        assert loss is not None
        with profile_range("backward"):
            loss.backward()

    if mode == "train_step":
        assert optimizer is not None
        with profile_range("optimizer"):
            optimizer.step()

    return loss, logits


def _summary(raw_timings_ms: list[float]) -> dict[str, float]:
    mean = statistics.mean(raw_timings_ms)
    sample_std = statistics.stdev(raw_timings_ms) if len(raw_timings_ms) > 1 else 0.0
    return {
        "mean_ms": mean,
        "sample_std_ms": sample_std,
        "cv": sample_std / mean if mean else 0.0,
        "min_ms": min(raw_timings_ms),
        "max_ms": max(raw_timings_ms),
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    command: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    _validate(config)
    device = resolve_device(config.device)
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    result: dict[str, Any] = {
        "status": "success",
        "experiment": "end_to_end_benchmark",
        "config": asdict(config),
        "model_config": model_config(config.model_size, config.context_length),
        "command": command,
        "environment": public_environment(device),
        "result_path": public_path(output_path) if output_path is not None else None,
        "timing": {
            "timer": "time.perf_counter",
            "cuda_synchronize_before_step": device.type == "cuda",
            "cuda_synchronize_after_step": device.type == "cuda",
            "raw_timings_ms": [],
        },
    }

    try:
        model = build_model(config.model_size, config.context_length, device)
        model.train(config.mode != "forward")
        optimizer = build_optimizer(model) if config.mode == "train_step" else None
        tokens = torch.randint(
            0,
            model_config(config.model_size, config.context_length)["vocab_size"],
            (config.batch_size, config.context_length),
            device=device,
        )
        targets = torch.randint_like(tokens, high=model_config(config.model_size, config.context_length)["vocab_size"])
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())

        losses: list[float] = []
        raw_timings_ms: list[float] = []
        for _ in range(config.warmup):
            execute_step(model, optimizer, tokens, targets, config.mode, config.dtype, device)
            synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(config.steps):
            synchronize(device)
            started = time.perf_counter()
            with profile_range("benchmark/measure_step"):
                loss, _ = execute_step(model, optimizer, tokens, targets, config.mode, config.dtype, device)
            synchronize(device)
            raw_timings_ms.append((time.perf_counter() - started) * 1000)
            if loss is not None:
                losses.append(float(loss.detach()))

        result["timing"]["raw_timings_ms"] = raw_timings_ms
        result["timing"].update(_summary(raw_timings_ms))
        result["loss_samples"] = losses
        result["memory"] = cuda_memory_metrics(device)
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["exception_type"] = type(exc).__name__
        result["error"] = sanitize_error(exc)
        result["memory"] = cuda_memory_metrics(device)
    except Exception as exc:
        result["status"] = "failed"
        result["exception_type"] = type(exc).__name__
        result["error"] = sanitize_error(exc)
        result["memory"] = cuda_memory_metrics(device)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P end-to-end benchmark")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=MODES, default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    config = BenchmarkConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        warmup=args.warmup,
        steps=args.steps,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
    )
    cli_arguments = list(arguments) if arguments is not None else sys.argv[1:]
    result = run_benchmark(
        config,
        command=public_command("profiling.benchmark", cli_arguments),
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"success", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
