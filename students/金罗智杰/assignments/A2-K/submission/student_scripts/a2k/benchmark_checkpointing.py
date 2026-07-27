"""Run the formal A2-K activation-checkpointing matrix on an RTX 4090."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k.checkpointing import transformer_forward_with_checkpointing
from student_scripts.a2k.common import (
    ALLOCATOR_LIMIT_MIB,
    HARD_LIMIT_MIB,
    configure_cuda_environment,
    environment_metadata,
    peak_memory,
    summarize_samples,
    timed_call,
    write_csv,
    write_json,
)

MODEL_CONFIG = {
    "vocab_size": 10_000,
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
    "rope_theta": 10_000.0,
}
STANDARD_BLOCK_SIZES = (None, 1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


def run_configuration(
    context_length: int,
    checkpoint_block_size: int | None,
    seed: int,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    config_id = f"medium_ctx{context_length}_block{checkpoint_block_size or 'none'}"
    row: dict[str, object] = {
        "config_id": config_id,
        "model_size": "medium",
        "num_layers": MODEL_CONFIG["num_layers"],
        "context_length": context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast",
        "checkpoint_block_size": checkpoint_block_size if checkpoint_block_size is not None else "none",
        "nested": False,
        "warmup_steps": warmup,
        "measurement_steps": steps,
    }

    model = optimizer = token_ids = targets = None
    try:
        model = BasicsTransformerLM(context_length=context_length, **MODEL_CONFIG).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        token_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (1, context_length), device=device)
        targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (1, context_length), device=device)

        def train_step() -> None:
            assert model is not None and optimizer is not None and token_ids is not None and targets is not None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = transformer_forward_with_checkpointing(model, token_ids, checkpoint_block_size)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()

        for _ in range(warmup):
            train_step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = [timed_call(train_step) for _ in range(steps)]
        allocated, reserved = peak_memory()
        summary = summarize_samples(samples)
        row.update(
            {
                "step_time_ms_samples": json.dumps([round(value, 6) for value in samples]),
                "step_time_ms_p50": summary["p50_ms"],
                "peak_allocated_mib": allocated,
                "peak_reserved_mib": reserved,
                "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
                "within_24gib": reserved <= HARD_LIMIT_MIB,
                "status": "success",
                "error": "",
            }
        )
    except torch.OutOfMemoryError as error:
        row.update(
            {
                "step_time_ms_samples": "[]",
                "step_time_ms_p50": "",
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
                "within_24gib": True,
                "status": "oom",
                "error": type(error).__name__,
            }
        )
    finally:
        model = optimizer = token_ids = targets = None
        gc.collect()
        torch.cuda.empty_cache()
    return row


def main() -> None:
    args = parse_args()
    environment = configure_cuda_environment(require_rtx4090=True)
    rows = [
        run_configuration(1024, block_size, args.seed, args.warmup, args.steps)
        for block_size in STANDARD_BLOCK_SIZES
    ]

    successful_checkpointed = [
        row for row in rows if row["status"] == "success" and row["checkpoint_block_size"] != "none"
    ]
    if not successful_checkpointed:
        raise RuntimeError("no checkpointed context-1024 configuration succeeded")
    best = min(successful_checkpointed, key=lambda row: float(row["peak_allocated_mib"]))
    best_block_size = int(best["checkpoint_block_size"])
    rows.extend(
        [
            run_configuration(2048, None, args.seed, args.warmup, args.steps),
            run_configuration(2048, best_block_size, args.seed, args.warmup, args.steps),
        ]
    )

    output_path = args.output_dir / "checkpointing.csv"
    write_csv(output_path, rows)
    metadata = environment_metadata(
        environment,
        command="python student_scripts/a2k/benchmark_checkpointing.py",
        seed=args.seed,
        warmup=args.warmup,
        measurement=args.steps,
    )
    metadata["selected_context_2048_checkpoint_block_size"] = best_block_size
    write_json(args.output_dir / "checkpointing.metadata.json", metadata)
    print(f"saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
