#!/usr/bin/env python3
"""One-process-per-case A2-K attention benchmark.

Formal cases are deliberately restricted to the handout matrix.  Every launch
measures exactly one implementation/shape/phase tuple, so an OOM or compiler
failure cannot contaminate a later case or trigger a silent smaller fallback.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

from student_scripts.a2k.runtime import (
    ALLOCATOR_LIMIT_MIB,
    RuntimeValidationError,
    assert_public_payload,
    atomic_write_json,
    peak_memory_mib,
    prepare_runtime,
    public_error,
    release_memory,
    reset_peak_memory,
    synchronize,
)

import torch


SCHEMA_VERSION = "cs336.a2k.attention-benchmark-case.v1"
IMPLEMENTATIONS = ("eager", "compiled", "triton")
PHASES = ("forward", "backward", "forward_backward")
CORE_SEQUENCE_LENGTHS = (512, 2048, 8192)
BOUNDARY_SEQUENCE_LENGTHS = (16384,)
HEAD_DIMS = (64, 128)
DO_BENCH_WARMUP_MS = 100
DO_BENCH_REP_MS = 300
QUANTILES = (0.2, 0.5, 0.8)


def official_cases() -> list[dict[str, Any]]:
    """Return the complete 54-row core plus 12-row boundary matrix."""

    rows: list[dict[str, Any]] = []
    for sequence_length in CORE_SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for implementation in IMPLEMENTATIONS:
                for phase in PHASES:
                    rows.append(
                        {
                            "case_id": case_id(sequence_length, head_dim, implementation, phase),
                            "matrix": "core",
                            "sequence_length": sequence_length,
                            "head_dim": head_dim,
                            "implementation": implementation,
                            "phase": phase,
                        }
                    )
    for sequence_length in BOUNDARY_SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for implementation in ("eager", "triton"):
                for phase in PHASES:
                    rows.append(
                        {
                            "case_id": case_id(sequence_length, head_dim, implementation, phase),
                            "matrix": "boundary",
                            "sequence_length": sequence_length,
                            "head_dim": head_dim,
                            "implementation": implementation,
                            "phase": phase,
                        }
                    )
    return rows


def case_id(sequence_length: int, head_dim: int, implementation: str, phase: str) -> str:
    return f"n{sequence_length:05d}-d{head_dim:03d}-{implementation}-{phase}"


def _matrix_name(sequence_length: int) -> str:
    return "boundary" if sequence_length in BOUNDARY_SEQUENCE_LENGTHS else "core"


def _is_official_case(sequence_length: int, head_dim: int, implementation: str, phase: str) -> bool:
    candidate = (sequence_length, head_dim, implementation, phase)
    return any(candidate == (row["sequence_length"], row["head_dim"], row["implementation"], row["phase"]) for row in official_cases())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one formal A2-K attention performance case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-cases", action="store_true", help="print the official 66-case manifest and exit")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a tiny CPU control-flow check; result is explicitly non-authoritative",
    )
    parser.add_argument(
        "--development-cuda",
        action="store_true",
        help="run the exact matrix under the 23 GiB cap on a non-standard GPU; explicitly non-authoritative",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_cases:
        return args
    if args.dry_run and args.development_cuda:
        parser.error("--dry-run and --development-cuda are mutually exclusive")
    missing = [
        flag
        for flag, value in (
            ("--sequence-length", args.sequence_length),
            ("--head-dim", args.head_dim),
            ("--implementation", args.implementation),
            ("--phase", args.phase),
            ("--output", args.output),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required unless --list-cases is used: {', '.join(missing)}")
    if not args.dry_run and not _is_official_case(args.sequence_length, args.head_dim, args.implementation, args.phase):
        parser.error("formal mode accepts only a case from --list-cases; shapes cannot be reduced")
    if args.dry_run and (args.sequence_length <= 0 or args.head_dim <= 0):
        parser.error("dry-run dimensions must be positive")
    return args


def _linear_quantile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _manual_cpu_bench(function: Callable[[], Any], *, grad_to_none: list[torch.Tensor] | None) -> dict[str, Any]:
    """Small control-flow timer; never described as equivalent GPU evidence."""

    if grad_to_none:
        for tensor in grad_to_none:
            tensor.grad = None
    value = function()
    del value
    samples: list[float] = []
    for _ in range(3):
        if grad_to_none:
            for tensor in grad_to_none:
                tensor.grad = None
        started = time.perf_counter()
        value = function()
        samples.append((time.perf_counter() - started) * 1000)
        del value
    return {
        "p20_ms": _linear_quantile(samples, 0.2),
        "p50_ms": statistics.median(samples),
        "p80_ms": _linear_quantile(samples, 0.8),
        "sample_count": len(samples),
        "private_raw_samples_retained": False,
    }


def _cuda_do_bench(
    function: Callable[[], Any],
    *,
    device: torch.device,
    grad_to_none: list[torch.Tensor] | None,
) -> dict[str, Any]:
    import triton

    synchronize(device)
    values = triton.testing.do_bench(
        function,
        warmup=DO_BENCH_WARMUP_MS,
        rep=DO_BENCH_REP_MS,
        grad_to_none=grad_to_none,
        quantiles=list(QUANTILES),
    )
    synchronize(device)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise RuntimeError("do_bench returned an unexpected quantile schema")
    p20, p50, p80 = (float(value) for value in values)
    if not all(math.isfinite(value) and value > 0 for value in (p20, p50, p80)):
        raise RuntimeError("do_bench returned a non-positive or non-finite latency")
    if not p20 <= p50 <= p80:
        raise RuntimeError("do_bench quantiles are not monotonic")
    return {
        "p20_ms": p20,
        "p50_ms": p50,
        "p80_ms": p80,
        "sample_count": None,
        "private_raw_samples_retained": False,
    }


def _load_operation(implementation: str, *, dry_run: bool) -> tuple[Callable[..., torch.Tensor] | None, dict[str, Any] | None]:
    # Importing this module happens only after prepare_runtime has applied the
    # allocator guard for formal CUDA execution.
    from cs336_systems.a2k.attention import FlashAttentionTriton, TRITON_CONFIG, eager_attention

    if implementation == "eager":
        return eager_attention, None
    if implementation == "compiled":
        return (
            torch.compile(
                eager_attention,
                backend="eager" if dry_run else "inductor",
                dynamic=False,
                fullgraph=not dry_run,
            ),
            None,
        )
    if implementation == "triton":
        if dry_run:
            return None, dict(TRITON_CONFIG)
        return (lambda q, k, v, is_causal: FlashAttentionTriton.apply(q, k, v, is_causal)), dict(TRITON_CONFIG)
    raise AssertionError(f"unhandled implementation: {implementation}")


def _seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _measure_case(
    *,
    operation: Callable[..., torch.Tensor],
    implementation: str,
    phase: str,
    sequence_length: int,
    head_dim: int,
    seed: int,
    device: torch.device,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    dtype = torch.float32 if dry_run else torch.bfloat16
    _seed(seed, device)
    shape = (1, sequence_length, head_dim)
    needs_grad = phase != "forward"
    query = torch.randn(shape, device=device, dtype=dtype, requires_grad=needs_grad)
    key = torch.randn(shape, device=device, dtype=dtype, requires_grad=needs_grad)
    value = torch.randn(shape, device=device, dtype=dtype, requires_grad=needs_grad)
    grad_output = torch.randn(shape, device=device, dtype=dtype) if needs_grad else None
    grad_inputs = [query, key, value] if needs_grad else None

    def forward() -> torch.Tensor:
        return operation(query, key, value, True)

    retained_output: torch.Tensor | None = None
    if phase == "forward":

        def measured() -> torch.Tensor:
            with torch.no_grad():
                return forward()

    elif phase == "backward":
        assert grad_output is not None
        retained_output = forward()
        synchronize(device)

        def measured() -> None:
            assert retained_output is not None
            retained_output.backward(grad_output, retain_graph=True)

    else:
        assert grad_output is not None

        def measured() -> None:
            output = forward()
            output.backward(grad_output)

    # Compile/autotune every kernel before resetting memory statistics.  The
    # required do_bench call still performs its own 100 ms warm-up, while the
    # reported peak now describes steady execution instead of compiler state.
    if not dry_run:
        if grad_inputs:
            for tensor in grad_inputs:
                tensor.grad = None
        warmup_value = measured()
        del warmup_value
        synchronize(device)
        if grad_inputs:
            for tensor in grad_inputs:
                tensor.grad = None
    reset_peak_memory(device)
    if dry_run:
        timings = _manual_cpu_bench(measured, grad_to_none=grad_inputs)
    else:
        timings = _cuda_do_bench(measured, device=device, grad_to_none=grad_inputs)
    memory = peak_memory_mib(device)
    retained_output = None
    query = key = value = grad_output = None
    release_memory(device)
    return timings, memory


def _base_record(args: argparse.Namespace) -> dict[str, Any]:
    dry_run = bool(args.dry_run)
    sequence_length = 16 if dry_run else int(args.sequence_length)
    head_dim = 32 if dry_run else int(args.head_dim)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "authoritative": False,
        "case_id": case_id(sequence_length, head_dim, str(args.implementation), str(args.phase)),
        "matrix": "dry_run" if dry_run else _matrix_name(sequence_length),
        "implementation": args.implementation,
        "batch_size": 1,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": "float32" if dry_run else "bfloat16",
        "causal": True,
        "phase": args.phase,
        "seed": args.seed,
        "timer": {
            "name": "time.perf_counter" if dry_run else "triton.testing.do_bench",
            "warmup_ms": None if dry_run else DO_BENCH_WARMUP_MS,
            "rep_ms": None if dry_run else DO_BENCH_REP_MS,
            "quantiles": list(QUANTILES),
            "synchronize_boundaries": not dry_run,
        },
        "compile": {
            "backend": ("eager" if dry_run else "inductor") if args.implementation == "compiled" else None,
            "dynamic": False if args.implementation == "compiled" else None,
            "fullgraph": (not dry_run) if args.implementation == "compiled" else None,
        },
        "measurement_boundary": ("forward graph prepared outside timing; retained graph backward only" if args.phase == "backward" else args.phase.replace("_", "+")),
        "input_creation_timed": False,
        "fallback_used": False,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _base_record(args)
    guard = None
    try:
        guard = prepare_runtime(
            dry_run=args.dry_run,
            tf32_enabled=False,
            development_cuda=args.development_cuda,
        )
        record["runtime"] = guard.metadata
        record["authoritative"] = guard.authoritative
        operation, triton_config = _load_operation(str(args.implementation), dry_run=args.dry_run)
        record["triton_config"] = triton_config if args.implementation == "triton" else None
        if operation is None:
            record.update(
                status="dry_run_skipped",
                non_authoritative_reason="student Triton kernel requires a real CUDA GPU; CPU dry-run did not execute it",
            )
            assert_public_payload(record)
            return record, 0

        timings, memory = _measure_case(
            operation=operation,
            implementation=str(args.implementation),
            phase=str(args.phase),
            sequence_length=int(record["sequence_length"]),
            head_dim=int(record["head_dim"]),
            seed=int(args.seed),
            device=guard.device,
            dry_run=bool(args.dry_run),
        )
        record["latency"] = timings
        record["memory"] = memory
        if guard.device.type == "cuda" and (memory["peak_reserved_mib"] is None or float(memory["peak_reserved_mib"]) > ALLOCATOR_LIMIT_MIB):
            raise RuntimeValidationError("peak reserved memory exceeded the 23 GiB allocator budget")
        record["status"] = "dry_run_ok" if args.dry_run else ("development_ok" if args.development_cuda else "ok")
        if args.dry_run:
            record["non_authoritative_reason"] = guard.metadata["non_authoritative_reason"]
        elif args.development_cuda:
            record["non_authoritative_reason"] = guard.metadata["non_authoritative_reason"]
        assert_public_payload(record)
        return record, 0
    except Exception as exc:
        record["status"] = "oom" if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower() else "invalid"
        record["error"] = public_error(exc)
        record["authoritative"] = bool(guard and guard.authoritative)
        if guard is not None:
            record.setdefault("runtime", guard.metadata)
            try:
                record["memory"] = peak_memory_mib(guard.device)
            except Exception:
                record["memory"] = {"peak_allocated_mib": None, "peak_reserved_mib": None}
            release_memory(guard.device)
        assert_public_payload(record)
        return record, 0 if record["status"] == "oom" else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        print(json.dumps(official_cases(), indent=2, sort_keys=True))
        return 0
    record, exit_code = run(args)
    atomic_write_json(args.output, record)
    print(f"{record['case_id']}: {record['status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
