"""Run one isolated activation-checkpointing configuration."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k.checkpointing import checkpoint_sequential_blocks
from student_scripts.a2k.common import make_metadata, parse_dtype, run_safe


MODEL_CONFIGS = {
    "medium": dict(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = MODEL_CONFIGS[args.model_size]
    command = " ".join(__import__("sys").argv)
    metadata = make_metadata(args.seed, command)
    metadata["experiment"] = {
        "model_size": args.model_size,
        "num_layers": config["num_layers"],
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": "bf16_autocast_fp32_params",
        "checkpoint_block_size": args.block_size or None,
        "nested": False,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
    }
    device = torch.device("cuda")
    dtype = parse_dtype("bf16")

    def body() -> dict[str, object]:
        torch.backends.cuda.matmul.allow_tf32 = False
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=args.context_length,
            **config,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tokens = torch.randint(
            0, 10_000, (args.batch_size, args.context_length), device=device
        )
        targets = torch.randint(
            0, 10_000, (args.batch_size, args.context_length), device=device
        )

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                embedded = model.token_embeddings(tokens)
                hidden = checkpoint_sequential_blocks(
                    tuple(model.layers),
                    embedded,
                    None if args.block_size <= 0 else args.block_size,
                )
                logits = model.lm_head(model.ln_final(hidden))
                loss = torch.nn.functional.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
            loss.backward()
            optimizer.step()

        for _ in range(args.warmup_steps):
            step()
        torch.cuda.synchronize()
        values: list[float] = []
        allocated_peaks: list[float] = []
        reserved_peaks: list[float] = []
        for _ in range(args.measurement_steps):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            step()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
            allocated_peaks.append(torch.cuda.max_memory_allocated() / 2**20)
            reserved_peaks.append(torch.cuda.max_memory_reserved() / 2**20)
        return {
            "config": {
                "model_size": args.model_size,
                "num_layers": config["num_layers"],
                "context_length": args.context_length,
                "batch_size": args.batch_size,
                "dtype": "bf16_autocast_fp32_params",
                "checkpoint_block_size": args.block_size or None,
                "nested": False,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
            },
            "step_time_ms_samples": values,
            "step_time_ms_p50": statistics.median(values),
            "peak_allocated_mib": max(allocated_peaks, default=0.0),
            "peak_reserved_mib": max(reserved_peaks, default=0.0),
        }

    return run_safe(args.output, metadata, body)


if __name__ == "__main__":
    raise SystemExit(main())
