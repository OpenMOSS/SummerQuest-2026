"""One-process activation-checkpointing training-step benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from .common import (
    configure_formal_run,
    public_run_record,
    upsert_csv_rows,
    upsert_json_record,
)

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k import CheckpointedTransformerLM
from cs336_systems.a2k.runtime import (
    classify_exception,
    peak_memory_mib,
    reset_peak_memory,
    timing_summary,
)


MEDIUM_CONFIG = {
    "vocab_size": 10_000,
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
    "rope_theta": 10_000.0,
}

FIELDS = (
    "config_id",
    "model_size",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "allocator_limit_mib",
    "allocator_fraction",
    "free_memory_mib_at_start",
    "final_loss",
    "status",
    "error_type",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument(
        "--checkpoint-block-size",
        type=int,
        choices=(0, 1, 2, 4, 8),
        required=True,
        help="0 selects the no-checkpoint baseline",
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = configure_formal_run(seed=args.seed, tf32_enabled=False)
    block_size = (
        None
        if args.checkpoint_block_size == 0
        else args.checkpoint_block_size
    )
    block_label = "none" if block_size is None else str(block_size)
    config_id = (
        f"medium-l24-b1-s{args.context_length}-bf16-block{block_label}"
    )
    row: dict[str, Any] = {
        "config_id": config_id,
        "model_size": "medium",
        "num_layers": 24,
        "context_length": args.context_length,
        "batch_size": 1,
        "dtype": "bfloat16_autocast_fp32_parameters",
        "checkpoint_block_size": block_label,
        "nested": False,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "step_time_ms_samples": "[]",
        "step_time_ms_p50": "",
        "allocator_limit_mib": run.allocator.allocator_limit_mib,
        "allocator_fraction": run.allocator.allocator_fraction,
        "free_memory_mib_at_start": run.free_memory_mib_at_start,
        "final_loss": "",
        "status": "ok",
        "error_type": "",
        "error": "",
    }
    try:
        base_model = BasicsTransformerLM(
            context_length=args.context_length,
            **MEDIUM_CONFIG,
        ).to("cuda")
        model = CheckpointedTransformerLM(base_model, block_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tokens = torch.randint(
            0,
            MEDIUM_CONFIG["vocab_size"],
            (1, args.context_length),
            device="cuda",
        )
        labels = torch.randint(
            0,
            MEDIUM_CONFIG["vocab_size"],
            (1, args.context_length),
            device="cuda",
        )

        def training_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(tokens)
            loss = functional.cross_entropy(
                logits.float().reshape(
                    -1,
                    MEDIUM_CONFIG["vocab_size"],
                ),
                labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            return float(loss.detach().item())

        final_loss = float("nan")
        for _ in range(args.warmup_steps):
            final_loss = training_step()
        torch.cuda.synchronize()
        reset_peak_memory()
        samples: list[float] = []
        for _ in range(args.measurement_steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            final_loss = training_step()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        summary = timing_summary(samples)
        row.update(
            {
                "step_time_ms_samples": json.dumps(samples),
                "step_time_ms_p50": summary["p50_ms"],
                "final_loss": final_loss,
                **peak_memory_mib(),
            }
        )
    except BaseException as error:
        row.update(classify_exception(error))
        try:
            row.update(peak_memory_mib())
        except RuntimeError:
            pass

    upsert_csv_rows(
        args.output,
        [row],
        key_fields=("config_id",),
        fieldnames=FIELDS,
    )
    command = (
        "python -m student_scripts.a2k.checkpoint_benchmark "
        f"--context-length {args.context_length} "
        f"--checkpoint-block-size {args.checkpoint_block_size} "
        f"--warmup-steps {args.warmup_steps} "
        f"--measurement-steps {args.measurement_steps} "
        f"--seed {args.seed}"
    )
    record = public_run_record(
        run=run,
        experiment="checkpointing",
        command=command,
        timer="CUDA events around one complete training step",
        warmup={"steps": args.warmup_steps},
        measurement={
            "steps": args.measurement_steps,
            "reported_quantile": 0.5,
        },
        extra={
            "config_id": config_id,
            "status": row["status"],
            "checkpoint_block_size": block_label,
            "nested": False,
        },
    )
    record["config_id"] = config_id
    upsert_json_record(
        args.metadata,
        record,
        key_fields=("experiment", "config_id"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
