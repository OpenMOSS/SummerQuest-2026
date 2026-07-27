from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from cs336_systems.a2k.flash import FlashAttentionTriton
from student_scripts.a2k.common import (
    append_csv,
    benchmark_quantiles,
    command_string,
    configure_formal_process,
    memory_peaks,
    public_environment,
    reset_peaks,
    write_json,
)

ATTENTION_FIELDS = [
    "implementation",
    "batch_size",
    "sequence_length",
    "head_dimension",
    "dtype",
    "causal",
    "phase",
    "warmup_ms",
    "rep_ms",
    "latency_ms_p20",
    "latency_ms_p50",
    "latency_ms_p80",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "compile_cold_start_ms",
    "query_tile",
    "key_tile",
    "num_warps",
    "num_stages",
    "status",
    "error_type",
    "command",
]


def make_function(implementation: str):
    if implementation == "eager":
        return lambda query, key, value: explicit_attention(query, key, value, True)
    if implementation == "compiled":
        return torch.compile(lambda query, key, value: explicit_attention(query, key, value, True), fullgraph=True)
    if implementation == "triton":
        return lambda query, key, value: FlashAttentionTriton.apply(query, key, value, True)
    raise ValueError(implementation)


def benchmark_phase(function, query, key, value, output_gradient, phase: str) -> tuple[float, float, float, float, float]:
    if phase == "forward":
        measured = lambda: function(query, key, value)
    elif phase == "backward":
        output = function(query, key, value)

        def measured():
            query.grad = None
            key.grad = None
            value.grad = None
            output.backward(output_gradient, retain_graph=True)

    elif phase == "forward_backward":

        def measured():
            query.grad = None
            key.grad = None
            value.grad = None
            function(query, key, value).backward(output_gradient)

    else:
        raise ValueError(phase)

    reset_peaks()
    p20, p50, p80 = benchmark_quantiles(measured)
    peak_allocated, peak_reserved = memory_peaks()
    return p20, p50, p80, peak_allocated, peak_reserved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one formal A2-K attention benchmark configuration.")
    parser.add_argument("--implementation", choices=["eager", "compiled", "triton"], required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dimension", type=int, choices=[64, 128], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fraction = configure_formal_process()
    environment = public_environment(fraction)
    torch.manual_seed(args.seed)
    base = {
        "implementation": args.implementation,
        "batch_size": 1,
        "sequence_length": args.sequence_length,
        "head_dimension": args.head_dimension,
        "dtype": "torch.bfloat16",
        "causal": True,
        "warmup_ms": 100,
        "rep_ms": 300,
        "compile_cold_start_ms": "",
        "query_tile": (32 if args.head_dimension >= 128 else 64) if args.implementation == "triton" else "",
        "key_tile": (32 if args.head_dimension >= 128 else 64) if args.implementation == "triton" else "",
        "num_warps": 4 if args.implementation == "triton" else "",
        "num_stages": (1 if args.head_dimension >= 128 else 2) if args.implementation == "triton" else "",
        "command": command_string(),
    }
    rows = []
    try:
        query = torch.randn(
            1,
            args.sequence_length,
            args.head_dimension,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.randn_like(query, requires_grad=True)
        value = torch.randn_like(query, requires_grad=True)
        output_gradient = torch.randn_like(query)
        function = make_function(args.implementation)
        if args.implementation == "compiled":
            torch.cuda.synchronize()
            start = time.perf_counter()
            cold_output = function(query, key, value)
            torch.cuda.synchronize()
            base["compile_cold_start_ms"] = (time.perf_counter() - start) * 1000
            del cold_output
        else:
            function(query, key, value)
            torch.cuda.synchronize()

        for phase in ("forward", "backward", "forward_backward"):
            p20, p50, p80, peak_allocated, peak_reserved = benchmark_phase(
                function,
                query,
                key,
                value,
                output_gradient,
                phase,
            )
            row = {
                **base,
                "phase": phase,
                "latency_ms_p20": p20,
                "latency_ms_p50": p50,
                "latency_ms_p80": p80,
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
                "status": "ok",
                "error_type": "",
            }
            append_csv(args.output, row, ATTENTION_FIELDS)
            rows.append(row)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        peak_allocated, peak_reserved = memory_peaks()
        for phase in ("forward", "backward", "forward_backward"):
            row = {
                **base,
                "phase": phase,
                "latency_ms_p20": "",
                "latency_ms_p50": "",
                "latency_ms_p80": "",
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
                "status": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                "error_type": type(exc).__name__,
            }
            append_csv(args.output, row, ATTENTION_FIELDS)
            rows.append(row)
    write_json(
        args.metadata,
        {
            "environment": environment,
            "seed": args.seed,
            "latest_configuration": {
                "implementation": args.implementation,
                "sequence_length": args.sequence_length,
                "head_dimension": args.head_dimension,
            },
            "latest_rows": rows,
        },
    )
    print([(row["phase"], row["status"], row["latency_ms_p50"]) for row in rows])


if __name__ == "__main__":
    main()
