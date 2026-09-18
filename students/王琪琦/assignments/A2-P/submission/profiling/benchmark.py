from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from profiling.common import MODELS, build_model, environment, step, sync, write_json


def run(args) -> dict:
    device = torch.device(args.device); torch.manual_seed(args.seed)
    model = build_model(args.model_size, args.context_length, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.mode == "train_step" else None
    tokens = torch.randint(0, 10000, (args.batch_size, args.context_length), device=device)
    targets = torch.randint_like(tokens, high=10000)
    for _ in range(args.warmup): step(model, optimizer, tokens, targets, args.mode, args.dtype, device); sync(device)
    torch.cuda.reset_peak_memory_stats(device); raw = []
    for _ in range(args.steps):
        sync(device); start = time.perf_counter(); step(model, optimizer, tokens, targets, args.mode, args.dtype, device); sync(device)
        raw.append((time.perf_counter() - start) * 1000)
    mean = statistics.mean(raw); std = statistics.stdev(raw) if len(raw) > 1 else 0.0
    return {"status": "success", "config": vars(args) | {"output": Path(args.output).name},
            "environment": environment(device), "timer": "time.perf_counter with CUDA sync before/after",
            "raw_timings_ms": raw, "mean_ms": mean, "sample_std_ms": std, "cv": std / mean,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-size",choices=MODELS,default="small"); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--context-length",type=int,default=512); p.add_argument("--mode",choices=("forward","forward_backward","train_step"),default="train_step"); p.add_argument("--warmup",type=int,default=5); p.add_argument("--steps",type=int,default=10); p.add_argument("--dtype",choices=("fp32","bf16"),default="fp32"); p.add_argument("--seed",type=int,default=2026); p.add_argument("--device",default="cuda"); p.add_argument("--output",required=True)
    a=p.parse_args(); result=run(a); write_json(a.output,result); print(result)
if __name__ == "__main__": main()
