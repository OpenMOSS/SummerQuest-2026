from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.checkpointing import transformer_lm_forward_with_checkpointing
from runtime import (
    MINIMUM_FREE_MIB,
    configure_single_gpu_allocator,
    peak_memory,
    synchronize,
)


MODEL_SIZE = "medium"
VOCAB_SIZE = 10_000
MODEL_CONFIG = {
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
}


def parse_checkpoint_block_size(value: str) -> int | None:
    if value == "none":
        return None
    block_size = int(value)
    if block_size <= 0:
        raise argparse.ArgumentTypeError("checkpoint block size must be positive or 'none'")
    return block_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one activation-checkpointing configuration.")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--checkpoint-block-size", type=parse_checkpoint_block_size, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-free-mib", type=float, default=MINIMUM_FREE_MIB)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_model(context_length: int) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        **MODEL_CONFIG,
    )
    return model.cuda().train()


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def train_step(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    checkpoint_block_size: int | None,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with autocast_context():
        logits = transformer_lm_forward_with_checkpointing(
            model,
            input_ids,
            checkpoint_block_size,
        )
        loss = cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    return loss.detach()


def write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.warmup < 3 or args.steps < 5:
        raise ValueError("formal checkpoint runs require at least 3 warm-up and 5 measurement steps")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    allocator, environment = configure_single_gpu_allocator(args.minimum_free_mib)

    command = f"uv run python {shlex.join(sys.argv)}"
    config = {
        "model_size": MODEL_SIZE,
        **MODEL_CONFIG,
        "vocab_size": VOCAB_SIZE,
        "context_length": args.context_length,
        "batch_size": 1,
        "dtype": "bf16",
        "parameter_dtype": "fp32",
        "optimizer": "AdamW",
        "checkpoint_block_size": args.checkpoint_block_size,
        "nested": False,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
    }
    result: dict[str, object] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": command,
        "config": config,
        "environment": environment,
        "allocator": allocator,
        "status": "running",
        "failure_stage": None,
        "error_type": None,
        "measurement_started": False,
    }

    failure_stage = "model_setup"
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        model = build_model(args.context_length)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (1, args.context_length),
            device="cuda",
        )
        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (1, args.context_length),
            device="cuda",
        )

        failure_stage = "warmup"
        for _ in range(args.warmup):
            train_step(
                model,
                optimizer,
                input_ids,
                targets,
                args.checkpoint_block_size,
            )
            synchronize()

        optimizer.zero_grad(set_to_none=True)
        synchronize()
        torch.cuda.reset_peak_memory_stats(0)

        failure_stage = "measurement"
        result["measurement_started"] = True
        samples_ms: list[float] = []
        losses: list[float] = []
        for _ in range(args.steps):
            synchronize()
            start = perf_counter()
            loss = train_step(
                model,
                optimizer,
                input_ids,
                targets,
                args.checkpoint_block_size,
            )
            synchronize()
            samples_ms.append((perf_counter() - start) * 1000)
            losses.append(loss.item())

        result.update(
            {
                "status": "ok",
                "timing": {
                    "timer": "time.perf_counter",
                    "synchronization": "torch.cuda.synchronize before and after each step",
                    "step_time_ms_samples": samples_ms,
                    "step_time_ms_p50": median(samples_ms),
                },
                "numerics": {
                    "losses": losses,
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                },
                "memory": {
                    "peak_scope": "measurement",
                    **peak_memory(),
                },
            }
        )
    except torch.OutOfMemoryError:
        result.update(
            {
                "status": "oom",
                "failure_stage": failure_stage,
                "error_type": "OutOfMemoryError",
                "memory": {
                    "peak_scope": "measurement" if result["measurement_started"] else "process_until_failure",
                    **peak_memory(),
                },
            }
        )

    write_result(args.output, result)


if __name__ == "__main__":
    main()
