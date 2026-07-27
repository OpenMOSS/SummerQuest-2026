from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from student_scripts.a2k.common import append_csv, peak_memory, quantiles_ms, require_cuda_and_limit_allocator
from student_scripts.a2k.model_utils import build_transformer

SMALL = {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("eager", "compiled"), required=True)
    parser.add_argument("--phase", choices=("forward", "forward_backward", "train_step"), required=True)
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/compile_comparison.csv"))
    args = parser.parse_args()
    device, _ = require_cuda_and_limit_allocator()
    torch.manual_seed(0)
    model = build_transformer(10_000, 512, SMALL).to(device)
    if args.implementation == "compiled":
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(10_000, (1, 512), device=device)
    targets = torch.randint(10_000, (1, 512), device=device)

    def operation():
        if args.phase == "forward":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(inputs)
            return
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        loss.backward()
        if args.phase == "train_step":
            optimizer.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - start) * 1000
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(10):
        torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    append_csv(args.output, {
        "scope": "language_model", "model_size": "small", "implementation": args.implementation,
        "batch_size": 1, "sequence_length": 512, "head_dim": "", "dtype": "bf16", "causal": True,
        "phase": args.phase, "cold_start_ms": cold_start_ms, **quantiles_ms(samples),
        "samples_ms": json.dumps(samples), **peak_memory(), "status": "success",
    })


if __name__ == "__main__":
    main()
