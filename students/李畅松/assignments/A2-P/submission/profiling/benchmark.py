from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from profiling.nvtx_ranges import nvtx_range


MODEL_CONFIGS = {
    "tiny": dict(d_model=64, num_layers=2, num_heads=4, d_ff=128),
    "small": dict(d_model=768, num_layers=12, num_heads=12, d_ff=3072),
    "medium": dict(d_model=1024, num_layers=24, num_heads=16, d_ff=4096),
    "large": dict(d_model=1280, num_layers=36, num_heads=20, d_ff=5120),
    "xl": dict(d_model=1600, num_layers=48, num_heads=25, d_ff=6400),
    "10b": dict(d_model=4096, num_layers=48, num_heads=32, d_ff=16384),
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = MODEL_CONFIGS[args.model_size]
    model = BasicsTransformerLM(args.vocab_size, args.context_length, **config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    amp_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    def step():
        if args.mode != "forward":
            optimizer.zero_grad(set_to_none=True)
        grad_context = torch.no_grad() if args.mode == "forward" else nullcontext()
        with nvtx_range("forward"), grad_context:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(inputs)
                loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        if args.mode != "forward":
            with nvtx_range("backward"):
                loss.backward()
        if args.mode == "train_step":
            with nvtx_range("optimizer"):
                optimizer.step()
        return float(loss.detach())

    with nvtx_range("profile/warmup"):
        for _ in range(args.warmup):
            step()
        synchronize(device)

    timings, losses = [], []
    with nvtx_range("profile/measure"):
        for _ in range(args.steps):
            synchronize(device)
            start = time.perf_counter()
            losses.append(step())
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)

    result = {
        "command": " ".join(sys.argv),
        "config": vars(args),
        "model_config": config,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda": torch.version.cuda,
        },
        "timings_ms": timings,
        "mean_ms": statistics.mean(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "cv": statistics.stdev(timings) / statistics.mean(timings) if len(timings) > 1 else 0.0,
        "last_loss": losses[-1],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=MODEL_CONFIGS, default="tiny")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="train_step")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--device")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/timings.json")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
