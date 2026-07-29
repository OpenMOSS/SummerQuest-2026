"""Run one activation-checkpointing configuration in a fresh process."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from student_scripts.a2k.common import (
    MODEL_CONFIGS,
    configure_single_gpu,
    peak_memory,
    public_gpu_metadata,
    synchronize,
    timing_summary,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument(
        "--checkpoint-block-size",
        choices=("none", "1", "2", "4", "8"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def model_forward(
    model: BasicsTransformerLM,
    tokens: torch.Tensor,
    block_size: int | None,
) -> torch.Tensor:
    hidden = model.token_embeddings(tokens)
    if block_size is None:
        for layer in model.layers:
            hidden = layer(hidden)
    else:
        for start in range(0, len(model.layers), block_size):
            end = min(start + block_size, len(model.layers))

            def run_group(value: torch.Tensor, begin: int = start, stop: int = end) -> torch.Tensor:
                for layer_index in range(begin, stop):
                    value = model.layers[layer_index](value)
                return value

            hidden = checkpoint(run_group, hidden, use_reentrant=False)
    return model.lm_head(model.ln_final(hidden))


def main() -> int:
    args = parse_args()
    allocator = configure_single_gpu()
    gpu = public_gpu_metadata()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    block_size = None if args.checkpoint_block_size == "none" else int(args.checkpoint_block_size)
    config_id = f"no_ckpt_ctx{args.context_length}" if block_size is None else f"ckpt_b{block_size}_ctx{args.context_length}"
    row = {
        "config_id": config_id,
        "model_size": "medium",
        "num_layers": 24,
        "context_length": args.context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_params",
        "checkpoint_block_size": "none" if block_size is None else block_size,
        "nested": False,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "step_time_ms_samples": "[]",
        "step_time_ms_p50": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "error",
        "error_type": "",
    }

    try:
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=args.context_length,
            **MODEL_CONFIGS["medium"],
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tokens = torch.randint(
            0,
            10_000,
            (1, args.context_length),
            device="cuda",
        )
        targets = torch.randint(
            0,
            10_000,
            (1, args.context_length),
            device="cuda",
        )

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model_forward(model, tokens, block_size)
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
            loss.backward()
            optimizer.step()

        for _ in range(args.warmup_steps):
            step()
            synchronize()

        samples: list[float] = []
        allocated_peaks: list[float] = []
        reserved_peaks: list[float] = []
        for _ in range(args.measurement_steps):
            torch.cuda.reset_peak_memory_stats()
            synchronize()
            start = time.perf_counter()
            step()
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            allocated, reserved = peak_memory()
            allocated_peaks.append(allocated)
            reserved_peaks.append(reserved)

        summary = timing_summary(samples)
        row.update(
            {
                "step_time_ms_samples": json.dumps(samples),
                "step_time_ms_p50": summary["p50_ms"],
                "peak_allocated_mib": max(allocated_peaks),
                "peak_reserved_mib": max(reserved_peaks),
                "status": "ok",
            }
        )
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        row["error_type"] = "OutOfMemoryError"
        try:
            allocated, reserved = peak_memory()
            row["peak_allocated_mib"] = allocated
            row["peak_reserved_mib"] = reserved
        except RuntimeError:
            pass
    except Exception as error:  # Preserve a reproducible failure row.
        row["error_type"] = type(error).__name__
        row["status"] = "error"

    payload = {
        "row": row,
        "allocator": allocator,
        "gpu": gpu,
        "command": (f"python -m student_scripts.a2k.checkpoint_benchmark --context-length {args.context_length} --checkpoint-block-size {args.checkpoint_block_size}"),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if row["status"] in {"ok", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
