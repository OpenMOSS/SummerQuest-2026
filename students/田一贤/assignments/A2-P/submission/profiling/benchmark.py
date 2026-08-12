"""End-to-end benchmark with explicit CUDA synchronization boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .common import (
    autocast_context,
    build_model,
    loss_from_logits,
    summarize,
    timed_step,
)
from .config import get_model_spec


def benchmark(args) -> dict:
    if not torch.cuda.is_available() and args.device == "cuda":
        return {**vars(args), "status": "not_run_no_cuda", "samples_ms": []}
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = build_model(args.model_size, args.context_length, device)
    spec = get_model_spec(args.model_size)
    tokens = torch.randint(
        0, spec.vocab_size, (args.batch_size, args.context_length), device=device
    )
    targets = torch.randint_like(tokens, high=spec.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        if args.mode == "forward":
            with torch.no_grad(), autocast_context(device, args.dtype):
                model(tokens)
            return
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            logits = model(tokens)
            loss = loss_from_logits(logits, targets)
        loss.backward()
        if args.mode == "train_step":
            optimizer.step()

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = [timed_step(step, device) for _ in range(args.steps)]
    stats = summarize(samples)
    stats.update(
        {
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "warmup": args.warmup,
            "steps": args.steps,
            "dtype": args.dtype,
            "seed": args.seed,
            "device": args.device,
            "timer": "perf_counter_ns",
            "synchronize_before_after": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda"
            else 0.0,
            "status": "pass",
        }
    )
    return stats


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = {k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()}
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(serial))
        if not exists:
            writer.writeheader()
        writer.writerow(serial)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-size", choices=("small", "medium", "large", "xl"), default="small"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        default="train_step",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.csv"))
    args = parser.parse_args()
    row = benchmark(args)
    write_csv(args.output, row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
