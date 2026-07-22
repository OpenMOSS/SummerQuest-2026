#!/usr/bin/env python3
"""Measure training throughput and peak memory across batch sizes until OOM."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.model import TransformerLM  # noqa: E402
from cs336_basics.training import AdamW, cross_entropy, get_batch, gradient_clipping  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite-config", type=Path, default=Path("configs/experiment_suite.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/batch_size/summary.jsonl"))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    with resolve(args.config).open(encoding="utf-8") as file:
        config = json.load(file)
    with resolve(args.suite_config).open(encoding="utf-8") as file:
        suite = json.load(file)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = config["training"].get("precision", "bf16")
    dtype = np.dtype(config["data"].get("dtype", "uint16"))
    train_path = resolve(Path(config["data"]["train_path"]))
    dataset = np.memmap(train_path, dtype=dtype, mode="r")
    context_length = int(config["model"]["context_length"])
    steps = int(suite["batch_size_sweep"]["steps_per_value"])
    batch_sizes = [int(value) for value in suite["batch_size_sweep"]["values"]]
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as log:
        for batch_size in batch_sizes:
            torch.manual_seed(int(config.get("seed", 42)))
            np.random.seed(int(config.get("seed", 42)))
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            model = None
            optimizer = None
            inputs = None
            targets = None
            logits = None
            loss = None
            try:
                model = TransformerLM(**config["model"]).to(device)
                optimizer = AdamW(
                    model.parameters(),
                    lr=float(config["training"]["max_lr"]),
                    betas=tuple(config["training"].get("betas", (0.9, 0.95))),
                    eps=float(config["training"].get("eps", 1e-8)),
                    weight_decay=float(config["training"].get("weight_decay", 0.1)),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                final_loss = float("nan")
                for step in range(steps):
                    inputs, targets = get_batch(dataset, batch_size, context_length, device)
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda" and precision != "fp32":
                        cast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
                        context = torch.autocast(device_type="cuda", dtype=cast_dtype)
                    else:
                        context = nullcontext()
                    with context:
                        logits = model(inputs)
                        loss = cross_entropy(logits.float(), targets)
                    loss.backward()
                    gradient_clipping(model.parameters(), float(config["training"].get("grad_clip", 1.0)))
                    optimizer.step()
                    final_loss = float(loss.detach())
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                processed_tokens = steps * batch_size * context_length
                metric = {
                    "status": "completed",
                    "batch_size": batch_size,
                    "step": steps,
                    "wall_clock_sec": elapsed,
                    "train_loss": final_loss,
                    "lr": float(config["training"]["max_lr"]),
                    "processed_tokens": processed_tokens,
                    "tokens_per_sec": processed_tokens / elapsed,
                    "peak_memory_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                }
            except torch.OutOfMemoryError:
                metric = {
                    "status": "oom",
                    "batch_size": batch_size,
                    "step": 0,
                    "wall_clock_sec": 0.0,
                    "train_loss": None,
                    "lr": float(config["training"]["max_lr"]),
                    "processed_tokens": 0,
                    "tokens_per_sec": None,
                    "peak_memory_bytes": (torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                }
            log.write(json.dumps(metric, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(metric, sort_keys=True), flush=True)
            inputs = targets = logits = loss = optimizer = model = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if metric["status"] == "oom":
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
