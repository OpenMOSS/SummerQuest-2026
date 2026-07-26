"""Benchmark the explicit, unfused PyTorch attention baseline."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from student_scripts.a2k.attention_measurement import measure_attention_phase
from student_scripts.a2k.common import (
    ALLOCATOR_LIMIT_MIB,
    HARD_LIMIT_MIB,
    configure_cuda_environment,
    environment_metadata,
    write_csv,
    write_json,
)

SEQUENCE_LENGTHS = (512, 2048, 8192)
HEAD_DIMS = (64, 128)
PHASES = ("forward", "backward", "forward_backward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = configure_cuda_environment(require_rtx4090=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows: list[dict[str, object]] = []

    for sequence_length in SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for phase in PHASES:
                q = k = v = None
                row: dict[str, object] = {
                    "implementation": "eager_pytorch",
                    "sequence_length": sequence_length,
                    "head_dim": head_dim,
                    "batch_size": 1,
                    "dtype": "bfloat16",
                    "is_causal": True,
                    "phase": phase,
                    "warmup_ms": args.warmup_ms,
                    "rep_ms": args.rep_ms,
                    "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
                }
                try:
                    q = torch.randn(
                        1,
                        sequence_length,
                        head_dim,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=phase != "forward",
                    )
                    k = torch.randn_like(q, requires_grad=phase != "forward")
                    v = torch.randn_like(q, requires_grad=phase != "forward")

                    def function(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                        return explicit_attention(q, k, v, is_causal=True)

                    measurement = measure_attention_phase(
                        function,
                        q,
                        k,
                        v,
                        phase,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                    )
                    row.update(measurement)
                    row["within_24gib"] = float(measurement["peak_reserved_mib"]) <= HARD_LIMIT_MIB
                    row["status"] = "success"
                    row["error"] = ""
                except torch.OutOfMemoryError as error:
                    row.update(
                        {
                            "latency_p20_ms": "",
                            "latency_p50_ms": "",
                            "latency_p80_ms": "",
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                            "within_24gib": True,
                            "status": "oom",
                            "error": type(error).__name__,
                        }
                    )
                rows.append(row)
                q = k = v = None
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"{sequence_length=} {head_dim=} {phase=} "
                    f"status={row['status']} p50_ms={row['latency_p50_ms']}"
                )

    output_path = args.output_dir / "attention_baseline.csv"
    write_csv(output_path, rows)
    write_json(
        args.output_dir / "attention_baseline.metadata.json",
        environment_metadata(
            environment,
            command="python student_scripts/a2k/benchmark_attention_baseline.py",
            seed=args.seed,
            warmup=f"{args.warmup_ms} ms",
            measurement=f"{args.rep_ms} ms",
        ),
    )
    print(f"saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
