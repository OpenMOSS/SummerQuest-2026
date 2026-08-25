from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from cs336_systems.a2k.checkpointing import checkpoint_blocks
from .common import configure_allocator_guard


def _model_forward(model, tokens, checkpoint_block_size: int):
    hidden = model.token_embeddings(tokens)
    if checkpoint_block_size == 0:
        for layer in model.layers:
            hidden = layer(hidden)
    else:
        hidden = checkpoint_blocks(model.layers, hidden, checkpoint_block_size)
    return model.lm_head(model.ln_final(hidden))


def _run_config(context_length: int, block_size: int, warmup: int, steps: int) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(20260811)
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=context_length,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, 10_000, (1, context_length), device=device)
    targets = torch.randint_like(tokens, high=10_000)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = _model_forward(model, tokens, block_size)
            loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten())
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        train_step()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        train_step()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "step_time_ms_samples": samples,
        "step_time_ms_p50": statistics.median(samples),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/checkpointing.csv")
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--include-2048", action="store_true")
    args = parser.parse_args()
    contexts = (1024, 2048) if args.include_2048 else (1024,)
    rows = []

    guard = configure_allocator_guard()
    for context_length in contexts:
        for block_size in (0, 1, 2, 4, 8):
            base = {
                "config_id": f"medium_24L_b1_c{context_length}_block{block_size}",
                "model_size": "medium",
                "num_layers": 24,
                "context_length": context_length,
                "batch_size": 1,
                "dtype": "bfloat16",
                "checkpoint_block_size": block_size,
                "nested": False,
                "warmup_steps": args.warmup,
                "measurement_steps": args.steps,
                "allocator_guard_applied": guard["applied"],
                "allocator_limit_mib": guard["limit_mib"],
                "allocator_fraction": guard["fraction"],
            }
            if not torch.cuda.is_available():
                rows.append({**base, "status": "not_run_no_cuda"})
                continue
            try:
                rows.append(
                    {
                        **base,
                        **_run_config(
                            context_length, block_size, args.warmup, args.steps
                        ),
                    }
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                rows.append(
                    {**base, "status": f"oom_or_runtime_error:{type(exc).__name__}"}
                )
            finally:
                gc.collect()
                torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if isinstance(row.get("step_time_ms_samples"), list):
                row["step_time_ms_samples"] = json.dumps(row["step_time_ms_samples"])
            writer.writerow(row)


if __name__ == "__main__":
    main()
