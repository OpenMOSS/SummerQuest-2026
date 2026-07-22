#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.model import TransformerLM
from cs336_basics.training import AdamW, cross_entropy


def run_worker(config_path: Path, batch_size: int) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    torch.manual_seed(config.get("seed", 42))
    model = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config.get("rope_theta", 10000.0),
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config["max_lr"], weight_decay=config.get("weight_decay", 0.1))
    tokens = torch.randint(
        0,
        config["vocab_size"],
        (batch_size, config["context_length"]),
        device=device,
    )
    targets = torch.randint(
        0,
        config["vocab_size"],
        (batch_size, config["context_length"]),
        device=device,
    )
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(tokens)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    result = {
        "batch_size": batch_size,
        "success": True,
        "step_time_sec": time.perf_counter() - start,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "loss": loss.item(),
    }
    print(json.dumps(result), flush=True)
    return 0


def run_controller(config: Path, candidates: list[int], output: Path) -> int:
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA-compatible device is visible")
    results = []
    for batch_size in candidates:
        command = [
            sys.executable,
            __file__,
            "--config",
            str(config),
            "--worker-batch",
            str(batch_size),
        ]
        process = subprocess.run(command, env=os.environ.copy(), text=True, capture_output=True, check=False)
        if process.returncode == 0:
            result = json.loads(process.stdout.strip().splitlines()[-1])
        else:
            result = {
                "batch_size": batch_size,
                "success": False,
                "returncode": process.returncode,
                "error": (process.stderr or process.stdout)[-2000:],
            }
        results.append(result)
        print(json.dumps(result), flush=True)
        if not result["success"]:
            break
    summary = {
        "device": torch.cuda.get_device_name(0),
        "config": str(config),
        "largest_successful_batch": max((r["batch_size"] for r in results if r["success"]), default=None),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure one-step memory use across batch sizes.")
    parser.add_argument("--config", type=Path, default=Path("configs/tinystories_baseline.json"))
    parser.add_argument("--candidates", type=int, nargs="+", default=[1, 16, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/batch_sizes.json"))
    parser.add_argument("--worker-batch", type=int)
    args = parser.parse_args()
    if args.worker_batch is not None:
        return run_worker(args.config, args.worker_batch)
    return run_controller(args.config, args.candidates, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
