"""Fixed eager/compiled/Triton attention performance and memory matrix."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch
import triton

from cs336_systems.a2k.attention import (
    FlashAttentionTriton,
    explicit_attention,
)
from student_scripts.a2k.common import (
    configure_single_gpu,
    peak_memory,
    public_gpu_metadata,
    write_csv,
    write_json,
)

WARMUP_MS = 100
REPETITION_MS = 300
QUANTILES = [0.2, 0.5, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--single",
        nargs=4,
        metavar=("IMPLEMENTATION", "SEQUENCE", "HEAD_DIM", "PHASE"),
    )
    parser.add_argument("--single-output", type=Path)
    return parser.parse_args()


def clear_inputs(*objects: object) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def make_action(
    attention: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    phase: str,
) -> Callable[[], torch.Tensor | None]:
    if phase == "forward":

        def forward() -> torch.Tensor:
            return attention(q, k, v, True)

        return forward

    if phase == "backward":
        output = attention(q, k, v, True)

        def backward() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            output.backward(grad_output, retain_graph=True)

        return backward

    if phase == "forward_backward":

        def forward_backward() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            output = attention(q, k, v, True)
            output.backward(grad_output)

        return forward_backward

    raise ValueError(f"unknown phase: {phase}")


def benchmark_row(
    implementation: str,
    sequence_length: int,
    head_dim: int,
    phase: str,
) -> dict:
    row = {
        "implementation": implementation,
        "batch_size": 1,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "is_causal": True,
        "phase": phase,
        "warmup_ms": WARMUP_MS,
        "rep_ms": REPETITION_MS,
        "p20_ms": "",
        "p50_ms": "",
        "p80_ms": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "speedup_vs_eager": "",
        "query_tile": 64 if head_dim <= 64 else 32,
        "key_tile": 64,
        "num_warps": 4,
        "num_stages": 2,
        "status": "error",
        "error_type": "",
    }
    if implementation != "triton":
        for key in ("query_tile", "key_tile", "num_warps", "num_stages"):
            row[key] = ""

    try:
        torch.manual_seed(2026)
        q = torch.randn(
            1,
            sequence_length,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        grad_output = torch.randn_like(q)
        if implementation == "eager":
            attention = explicit_attention
        elif implementation == "compiled":
            attention = torch.compile(explicit_attention, backend="inductor", fullgraph=True)
        elif implementation == "triton":
            attention = FlashAttentionTriton.apply
        else:
            raise ValueError(implementation)

        action = make_action(attention, q, k, v, grad_output, phase)
        # Preflight catches compilation and lazy setup outside the formal timing.
        action()
        torch.cuda.synchronize()
        q.grad = None
        k.grad = None
        v.grad = None

        measured = triton.testing.do_bench(
            action,
            warmup=WARMUP_MS,
            rep=REPETITION_MS,
            quantiles=QUANTILES,
        )
        if not isinstance(measured, list):
            measured = [measured]

        torch.cuda.reset_peak_memory_stats()
        action()
        torch.cuda.synchronize()
        allocated, reserved = peak_memory()
        row.update(
            {
                "p20_ms": float(measured[0]),
                "p50_ms": float(measured[1]),
                "p80_ms": float(measured[2]),
                "peak_allocated_mib": allocated,
                "peak_reserved_mib": reserved,
                "status": "ok",
            }
        )
        clear_inputs(q, k, v, grad_output, action, attention)
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        row["error_type"] = "OutOfMemoryError"
        try:
            allocated, reserved = peak_memory()
            row["peak_allocated_mib"] = allocated
            row["peak_reserved_mib"] = reserved
        except RuntimeError:
            pass
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as error:
        row["status"] = "error"
        row["error_type"] = type(error).__name__
        gc.collect()
        torch.cuda.empty_cache()
    return row


def add_speedups(rows: list[dict]) -> None:
    eager = {
        (
            row["sequence_length"],
            row["head_dim"],
            row["phase"],
        ): float(row["p50_ms"])
        for row in rows
        if row["implementation"] == "eager" and row["status"] == "ok"
    }
    for row in rows:
        key = (row["sequence_length"], row["head_dim"], row["phase"])
        if row["status"] == "ok" and key in eager:
            row["speedup_vs_eager"] = eager[key] / float(row["p50_ms"])


def retry_in_fresh_process(row: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="a2k-attention-retry-") as temporary:
        output = Path(temporary) / "row.json"
        command = [
            sys.executable,
            "-m",
            "student_scripts.a2k.run_attention_benchmarks",
            "--flash-output",
            str(output),
            "--baseline-output",
            str(output),
            "--single",
            row["implementation"],
            str(row["sequence_length"]),
            str(row["head_dim"]),
            row["phase"],
            "--single-output",
            str(output),
        ]
        environment = os.environ.copy()
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        subprocess.run(command, env=environment, check=True)
        from student_scripts.a2k.common import read_json

        return read_json(output)["row"]


def main() -> int:
    args = parse_args()
    allocator = configure_single_gpu()
    gpu = public_gpu_metadata()
    torch.backends.cuda.matmul.allow_tf32 = True

    if args.single:
        if args.single_output is None:
            raise ValueError("--single-output is required with --single")
        implementation, sequence, head_dim, phase = args.single
        row = benchmark_row(implementation, int(sequence), int(head_dim), phase)
        write_json(args.single_output, {"row": row})
        return 0

    rows: list[dict] = []
    for sequence_length in (512, 2048, 8192):
        for head_dim in (64, 128):
            for phase in ("forward", "backward", "forward_backward"):
                for implementation in ("eager", "compiled", "triton"):
                    row = benchmark_row(implementation, sequence_length, head_dim, phase)
                    rows.append(row)
                    print(
                        implementation,
                        sequence_length,
                        head_dim,
                        phase,
                        row["status"],
                        row["p50_ms"],
                        flush=True,
                    )

    for head_dim in (64, 128):
        for phase in ("forward", "backward", "forward_backward"):
            for implementation in ("eager", "triton"):
                row = benchmark_row(implementation, 16384, head_dim, phase)
                rows.append(row)
                print(
                    implementation,
                    16384,
                    head_dim,
                    phase,
                    row["status"],
                    row["p50_ms"],
                    flush=True,
                )

    for index, row in enumerate(rows):
        if row["status"] == "error":
            retried = retry_in_fresh_process(row)
            if retried["status"] == "ok":
                rows[index] = retried

    add_speedups(rows)
    write_csv(args.flash_output, rows)
    baseline_rows = [
        {
            "batch_size": row["batch_size"],
            "sequence_length": row["sequence_length"],
            "head_dim": row["head_dim"],
            "dtype": row["dtype"],
            "is_causal": row["is_causal"],
            "phase": row["phase"],
            "warmup_ms": row["warmup_ms"],
            "rep_ms": row["rep_ms"],
            "p20_ms": row["p20_ms"],
            "p50_ms": row["p50_ms"],
            "p80_ms": row["p80_ms"],
            "peak_allocated_mib": row["peak_allocated_mib"],
            "peak_reserved_mib": row["peak_reserved_mib"],
            "status": row["status"],
            "error_type": row["error_type"],
        }
        for row in rows
        if row["implementation"] == "eager" and row["sequence_length"] in (512, 2048, 8192)
    ]
    write_csv(args.baseline_output, baseline_rows)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "experiment": "attention_performance",
                "allocator": allocator,
                "gpu": gpu,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "triton_version": triton.__version__,
                "timer": "triton.testing.do_bench",
                "warmup_ms": WARMUP_MS,
                "rep_ms": REPETITION_MS,
                "quantiles": QUANTILES,
                "tf32_enabled": True,
                "command": ("python -m student_scripts.a2k.run_attention_benchmarks --flash-output results/flash_benchmark.csv --baseline-output results/attention_baseline.csv"),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
