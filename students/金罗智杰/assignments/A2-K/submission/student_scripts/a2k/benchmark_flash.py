"""Run the fixed eager/compiled/Triton FlashAttention performance matrix."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch
import torch._functorch.config as functorch_config

from cs336_systems.a2k.attention import FlashAttentionTriton, explicit_attention
from student_scripts.a2k.attention_measurement import measure_attention_phase
from student_scripts.a2k.common import (
    ALLOCATOR_LIMIT_MIB,
    HARD_LIMIT_MIB,
    configure_cuda_environment,
    environment_metadata,
    write_csv,
    write_json,
)

CORE_SEQUENCE_LENGTHS = (512, 2048, 8192)
BOUNDARY_SEQUENCE_LENGTH = 16384
HEAD_DIMS = (64, 128)
PHASES = ("forward", "backward", "forward_backward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    return parser.parse_args()


def function_for(implementation: str):
    def eager(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return explicit_attention(q, k, v, is_causal=True)

    if implementation == "eager_pytorch":
        return eager
    if implementation == "compiled_pytorch":
        return torch.compile(eager, fullgraph=True)
    if implementation == "triton":
        return lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True)
    raise ValueError(f"unknown implementation: {implementation}")


def run_row(
    implementation: str,
    sequence_length: int,
    head_dim: int,
    phase: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query_tile_size = 64 if implementation == "triton" else ""
    key_tile_size = (64 if head_dim <= 64 else 32) if implementation == "triton" else ""
    row: dict[str, Any] = {
        "implementation": implementation,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "batch_size": 1,
        "dtype": "bfloat16",
        "is_causal": True,
        "phase": phase,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "cold_start_ms": "",
        "latency_p20_ms": "",
        "latency_p50_ms": "",
        "latency_p80_ms": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "speedup_vs_eager": "",
        "query_tile_size": query_tile_size,
        "key_tile_size": key_tile_size,
        "num_warps": 4 if implementation == "triton" else "",
        "num_stages": 2 if implementation == "triton" else "",
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "within_24gib": "",
        "status": "",
        "error": "",
    }
    q = k = v = None
    try:
        requires_grad = phase != "forward"
        q = torch.randn(
            1,
            sequence_length,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=requires_grad,
        )
        k = torch.randn_like(q, requires_grad=requires_grad)
        v = torch.randn_like(q, requires_grad=requires_grad)
        measurement = measure_attention_phase(
            function_for(implementation),
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
    except Exception as error:
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        summary = " ".join(str(error).split())[:300]
        row.update(
            {
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": reserved,
                "within_24gib": reserved <= HARD_LIMIT_MIB,
                "status": "oom" if isinstance(error, torch.OutOfMemoryError) else "error",
                "error": f"{type(error).__name__}: {summary}",
            }
        )
    finally:
        q = k = v = None
        gc.collect()
        torch.cuda.empty_cache()
    return row


def add_speedups(rows: list[dict[str, Any]]) -> None:
    eager_p50 = {
        (row["sequence_length"], row["head_dim"], row["phase"]): float(row["latency_p50_ms"])
        for row in rows
        if row["implementation"] == "eager_pytorch" and row["status"] == "success"
    }
    for row in rows:
        key = (row["sequence_length"], row["head_dim"], row["phase"])
        if row["status"] == "success" and key in eager_p50:
            row["speedup_vs_eager"] = eager_p50[key] / float(row["latency_p50_ms"])


def main() -> None:
    args = parse_args()
    # Required for repeated backward-only timing with retain_graph=True.
    functorch_config.donated_buffer = False
    environment = configure_cuda_environment(require_rtx4090=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows: list[dict[str, Any]] = []

    for sequence_length in CORE_SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for phase in PHASES:
                for implementation in ("eager_pytorch", "compiled_pytorch", "triton"):
                    row = run_row(implementation, sequence_length, head_dim, phase, args)
                    rows.append(row)
                    print(
                        f"core {implementation=} {sequence_length=} {head_dim=} {phase=} "
                        f"status={row['status']}"
                    )

    for head_dim in HEAD_DIMS:
        for phase in PHASES:
            for implementation in ("eager_pytorch", "triton"):
                row = run_row(implementation, BOUNDARY_SEQUENCE_LENGTH, head_dim, phase, args)
                rows.append(row)
                print(
                    f"boundary {implementation=} sequence_length={BOUNDARY_SEQUENCE_LENGTH} "
                    f"{head_dim=} {phase=} status={row['status']}"
                )

    add_speedups(rows)
    output_path = args.output_dir / "flash_benchmark.csv"
    write_csv(output_path, rows)
    metadata = environment_metadata(
        environment,
        command="python student_scripts/a2k/benchmark_flash.py",
        seed=args.seed,
        warmup=f"{args.warmup_ms} ms",
        measurement=f"{args.rep_ms} ms",
    )
    metadata["compile_config"] = {"torch_functorch_donated_buffer": False}
    write_json(args.output_dir / "flash_benchmark.metadata.json", metadata)
    print(f"saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
