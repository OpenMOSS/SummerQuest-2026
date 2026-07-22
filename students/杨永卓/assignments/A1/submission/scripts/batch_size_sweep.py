#!/usr/bin/env python3
"""Measure feasible batch sizes, throughput, and peak CUDA memory."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from cs336_basics.experiment import build_model, load_json, save_json, set_seed
from cs336_basics.training import AdamW, cross_entropy, get_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 640],
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    set_seed(config.get("seed", 42))
    device = torch.device("cuda")
    data = np.memmap(config["data"]["train"], dtype=np.uint16, mode="r")
    results = []
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_model(config, device)
        optimizer = AdamW(model.parameters(), lr=config["training"]["max_lr"], weight_decay=0.1)
        started = time.perf_counter()
        status = "ok"
        try:
            for _ in range(args.steps):
                x, y = get_batch(data, batch_size, config["model"]["context_length"], device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(x)
                    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            status = "oom"
            loss = torch.tensor(float("nan"))
        elapsed = time.perf_counter() - started
        results.append(
            {
                "batch_size": batch_size,
                "status": status,
                "steps": args.steps,
                "wall_clock_sec": elapsed,
                "tokens_per_sec": batch_size * config["model"]["context_length"] * args.steps / elapsed,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "last_loss": float(loss.item()),
            }
        )
        del model, optimizer
        torch.cuda.empty_cache()
        if status == "oom":
            break
    save_json(args.output, {"results": results})
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
