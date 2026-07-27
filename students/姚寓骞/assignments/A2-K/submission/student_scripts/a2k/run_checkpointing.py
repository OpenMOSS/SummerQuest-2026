from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_systems.a2k.checkpointing import run_checkpointed_blocks
from student_scripts.a2k.common import append_csv, peak_memory, require_cuda_and_limit_allocator
from student_scripts.a2k.model_utils import build_transformer

MEDIUM = {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096}


def forward(model, tokens, block_size):
    hidden = model.token_embeddings(tokens)
    hidden = run_checkpointed_blocks(model.layers, hidden, block_size)
    return model.lm_head(model.ln_final(hidden))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, choices=(1024, 2048), required=True)
    parser.add_argument("--checkpoint-block-size", choices=("none", "1", "2", "4", "8"), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/checkpointing.csv"))
    args = parser.parse_args()
    device, _ = require_cuda_and_limit_allocator()
    block_size = None if args.checkpoint_block_size == "none" else int(args.checkpoint_block_size)
    torch.manual_seed(0)
    model = build_transformer(10_000, args.context_length, MEDIUM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(10_000, (1, args.context_length), device=device)
    targets = torch.randint(10_000, (1, args.context_length), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward(model, tokens, block_size)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        loss.backward()
        optimizer.step()

    status, error, samples = "success", "", []
    try:
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        memory = peak_memory()
    except torch.OutOfMemoryError as exc:
        status, error, memory = "oom", type(exc).__name__, peak_memory()
    row = {
        "config_id": f"medium_c{args.context_length}_block_{args.checkpoint_block_size}", "model_size": "medium",
        "num_layers": 24, "context_length": args.context_length, "batch_size": 1, "dtype": "bf16_autocast_fp32_params",
        "checkpoint_block_size": args.checkpoint_block_size, "nested": False, "warmup_steps": args.warmup,
        "measurement_steps": args.steps, "step_time_ms_samples": json.dumps(samples),
        "step_time_ms_p50": statistics.median(samples) if samples else "", **memory, "status": status, "error_type": error,
    }
    append_csv(args.output, row)


if __name__ == "__main__":
    main()
