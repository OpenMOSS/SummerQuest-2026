"""One-process explicit PyTorch attention benchmark row."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from .common import (
    configure_formal_run,
    public_run_record,
    upsert_csv_rows,
    upsert_json_record,
)
from .attention_utils import (
    Phase,
    benchmark_attention_phase,
    make_attention_inputs,
)

from cs336_systems.a2k import explicit_attention
from cs336_systems.a2k.runtime import classify_exception, peak_memory_mib


FIELDS = (
    "implementation",
    "seq_len",
    "head_dim",
    "batch_size",
    "dtype",
    "causal",
    "phase",
    "warmup_ms",
    "rep_ms",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "sample_count",
    "measurement_elapsed_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "allocator_limit_mib",
    "allocator_fraction",
    "free_memory_mib_at_start",
    "status",
    "error_type",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument(
        "--phase",
        choices=("forward", "backward", "forward-backward"),
        required=True,
    )
    parser.add_argument("--warmup-ms", type=float, default=100.0)
    parser.add_argument("--rep-ms", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    phase: Phase = args.phase
    run = configure_formal_run(seed=args.seed, tf32_enabled=False)
    config_id = f"eager-s{args.seq_len}-d{args.head_dim}-{phase}"
    row: dict[str, Any] = {
        "implementation": "eager",
        "seq_len": args.seq_len,
        "head_dim": args.head_dim,
        "batch_size": 1,
        "dtype": "bfloat16",
        "causal": True,
        "phase": phase,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "allocator_limit_mib": run.allocator.allocator_limit_mib,
        "allocator_fraction": run.allocator.allocator_fraction,
        "free_memory_mib_at_start": run.free_memory_mib_at_start,
        "status": "ok",
        "error_type": "",
        "error": "",
    }
    try:
        inputs = make_attention_inputs(
            sequence_length=args.seq_len,
            head_dim=args.head_dim,
            phase=phase,
            seed=args.seed,
        )

        def forward() -> torch.Tensor:
            return explicit_attention(
                inputs.q,
                inputs.k,
                inputs.v,
                True,
            )

        row.update(
            benchmark_attention_phase(
                forward,
                inputs,
                phase,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
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
        key_fields=("implementation", "seq_len", "head_dim", "phase"),
        fieldnames=FIELDS,
    )
    command = (
        "python -m student_scripts.a2k.attention_baseline "
        f"--seq-len {args.seq_len} --head-dim {args.head_dim} "
        f"--phase {phase} --warmup-ms {args.warmup_ms:g} "
        f"--rep-ms {args.rep_ms:g} --seed {args.seed}"
    )
    record = public_run_record(
        run=run,
        experiment="attention_baseline",
        command=command,
        timer="CUDA events; prepare excluded from backward-only latency",
        warmup={"milliseconds": args.warmup_ms},
        measurement={
            "milliseconds": args.rep_ms,
            "quantiles": [0.2, 0.5, 0.8],
        },
        extra={
            "config_id": config_id,
            "status": row["status"],
            "phase": phase,
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
