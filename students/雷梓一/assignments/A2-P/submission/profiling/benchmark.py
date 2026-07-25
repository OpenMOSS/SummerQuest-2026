from __future__ import annotations

import argparse
import math
import statistics
import timeit
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from .common import (
    MODEL_CONFIGS,
    autocast_context,
    command_string,
    cuda_memory_metrics,
    model_config_dict,
    range_context,
    set_seed,
    software_metadata,
    utc_now,
    write_json,
)
from .nvtx_ranges import annotated_attention


MODES = ("forward", "forward_backward", "train_step")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A2-P synchronized end-to-end benchmark")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--mode", choices=MODES, default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def build_workload(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Tensor, torch.Tensor, torch.device]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size <= 0 or args.context_length <= 0 or args.steps <= 0 or args.warmup < 0:
        raise ValueError("batch-size, context-length, and steps must be positive; warmup must be non-negative")

    set_seed(args.seed)
    config = MODEL_CONFIGS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    tokens = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    return model, optimizer, tokens, targets, device


def make_step(
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> Callable[[], float | None]:
    def loss_for(logits: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits.flatten(0, 1).float(), targets.flatten())

    def forward() -> None:
        with torch.no_grad(), range_context("forward", device), autocast_context(device, args.dtype):
            model(tokens)
        return None

    def forward_backward() -> float:
        optimizer.zero_grad(set_to_none=True)
        with range_context("forward", device), autocast_context(device, args.dtype):
            logits = model(tokens)
            loss = loss_for(logits)
        with range_context("backward", device):
            loss.backward()
        return float(loss.detach())

    def train_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        with range_context("forward", device), autocast_context(device, args.dtype):
            logits = model(tokens)
            loss = loss_for(logits)
        with range_context("backward", device):
            loss.backward()
        with range_context("optimizer", device):
            optimizer.step()
        return float(loss.detach())

    return {"forward": forward, "forward_backward": forward_backward, "train_step": train_step}[args.mode]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def execute_steps(step: Callable[[], float | None], count: int, device: torch.device, range_name: str) -> list[float | None]:
    values = []
    with range_context(range_name, device):
        for _ in range(count):
            values.append(step())
            synchronize(device)
    return values


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    model, optimizer, tokens, targets, device = build_workload(args)
    step = make_step(args, model, optimizer, tokens, targets, device)

    with annotated_attention():
        execute_steps(step, args.warmup, device, "profile/warmup")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        raw_seconds: list[float] = []
        losses: list[float | None] = []
        with range_context("profile/measure", device):
            for _ in range(args.steps):
                synchronize(device)
                started = timeit.default_timer()
                losses.append(step())
                synchronize(device)
                raw_seconds.append(timeit.default_timer() - started)

    mean_seconds = statistics.fmean(raw_seconds)
    std_seconds = statistics.stdev(raw_seconds) if len(raw_seconds) > 1 else 0.0
    result = {
        "schema_version": 1,
        "status": "ok",
        "timestamp_utc": utc_now(),
        "command": command_string(),
        "result_file": args.output.name,
        "config": {
            "model_size": args.model_size,
            **model_config_dict(args.model_size),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "mode": args.mode,
            "warmup": args.warmup,
            "steps": args.steps,
            "dtype": args.dtype,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
        },
        "raw_seconds": raw_seconds,
        "mean_seconds": mean_seconds,
        "sample_std_seconds": std_seconds,
        "cv": std_seconds / mean_seconds if mean_seconds else math.nan,
        "losses": losses,
        "memory": cuda_memory_metrics(device),
        "software": software_metadata(device),
    }
    write_json(args.output, result)
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_benchmark(args)
    print(f"wrote {args.output}")
    print(
        f"{args.mode}: mean={result['mean_seconds']:.6f}s "
        f"std={result['sample_std_seconds']:.6f}s cv={result['cv']:.4f}"
    )


if __name__ == "__main__":
    main()
