from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "cs336-basics"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.checkpointing import CheckpointedBasicsTransformerLM


MEDIUM_CONFIG = {
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
}
ALLOCATOR_LIMIT_MIB = 23 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-K activation checkpointing measurement")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--checkpoint-block-size", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_allocator_limit() -> float:
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    limit_bytes = ALLOCATOR_LIMIT_MIB * 1024**2
    fraction = min(1.0, limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def run_training_step(model, optimizer, tokens, targets) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(tokens)
        loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.context_length <= 0 or args.batch_size <= 0:
        raise ValueError("context length and batch size must be positive")
    if args.checkpoint_block_size < 0:
        raise ValueError("checkpoint block size must be non-negative")

    allocator_fraction = configure_allocator_limit()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    block_size = args.checkpoint_block_size or None
    row = {
        "config_id": f"c{args.context_length}_b{block_size or 0}",
        "model_size": "medium",
        "num_layers": MEDIUM_CONFIG["num_layers"],
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": "bf16-autocast/fp32-params",
        "checkpoint_block_size": block_size or 0,
        "nested": False,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": allocator_fraction,
        "status": "ok",
    }

    try:
        model = CheckpointedBasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            checkpoint_block_size=block_size,
            rope_theta=10_000.0,
            **MEDIUM_CONFIG,
        ).to(device="cuda", dtype=torch.float32)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device="cuda")

        for _ in range(args.warmup_steps):
            run_training_step(model, optimizer, tokens, targets)
        torch.cuda.synchronize()

        latency_ms = []
        allocated_mib = []
        reserved_mib = []
        for _ in range(args.measurement_steps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_training_step(model, optimizer, tokens, targets)
            torch.cuda.synchronize()
            latency_ms.append((time.perf_counter() - start) * 1_000)
            allocated_mib.append(torch.cuda.max_memory_allocated() / 1024**2)
            reserved_mib.append(torch.cuda.max_memory_reserved() / 1024**2)

        row.update(
            {
                "step_time_ms_samples": json.dumps(latency_ms),
                "step_time_ms_p50": statistics.median(latency_ms),
                "peak_allocated_mib": max(allocated_mib),
                "peak_reserved_mib": max(reserved_mib),
            }
        )
    except torch.OutOfMemoryError as exc:
        row.update(
            {
                "status": "oom",
                "error": str(exc),
                "step_time_ms_samples": "[]",
                "step_time_ms_p50": "",
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        )
    finally:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)

    print(json.dumps(row, ensure_ascii=False))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
