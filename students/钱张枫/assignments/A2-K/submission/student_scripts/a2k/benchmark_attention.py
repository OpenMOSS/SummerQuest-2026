"""Measure the required explicit eager PyTorch-attention baseline matrix.

The baseline intentionally routes through ``cs336_systems.a2k.attention`` and
therefore materializes ``QK^T``; it never calls a fused attention operator.
Inputs are allocated before the timed region, and every OOM remains a CSV row.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

try:  # Support `python -m` and direct execution.
    from .common import (
        CudaPreflightError,
        PhaseMeasurement,
        append_memory_observation,
        append_run_metadata,
        cleanup_cuda,
        configure_cuda,
        current_peak_memory_mib,
        default_output_dir,
        dtype_name,
        error_kind,
        explicit_attention_apply,
        is_out_of_memory,
        make_attention_inputs,
        make_attention_phase,
        maximum_peak,
        measurement_csv_fields,
        measure_cuda_workload,
        parse_positive_ints,
        record_preflight_failure,
        stderr,
        write_csv,
    )
except ImportError:  # pragma: no cover - direct-script fallback.
    from common import (  # type: ignore[no-redef]
        CudaPreflightError,
        PhaseMeasurement,
        append_memory_observation,
        append_run_metadata,
        cleanup_cuda,
        configure_cuda,
        current_peak_memory_mib,
        default_output_dir,
        dtype_name,
        error_kind,
        explicit_attention_apply,
        is_out_of_memory,
        make_attention_inputs,
        make_attention_phase,
        maximum_peak,
        measurement_csv_fields,
        measure_cuda_workload,
        parse_positive_ints,
        record_preflight_failure,
        stderr,
        write_csv,
    )


PHASES: tuple[str, ...] = ("forward", "backward", "forward_backward")
RESULT_FIELDS: tuple[str, ...] = (
    "implementation",
    "batch_size",
    "sequence_length",
    "head_dim",
    "dtype",
    "is_causal",
    "phase",
    "seed",
    "formal",
    "warmup_ms",
    "rep_ms",
    "timer",
    "measurement_sample_count",
    "measurement_duration_ms",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "allocator_limit_mib",
    "allocator_fraction",
    "status",
    "error_kind",
    "reason",
)


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    batch_size: int
    sequence_lengths: tuple[int, ...]
    head_dims: tuple[int, ...]
    seed: int
    warmup_ms: int
    rep_ms: int
    formal: bool

    def as_json(self) -> dict[str, object]:
        return {
            "implementation": "eager_explicit_pytorch",
            "batch_size": self.batch_size,
            "sequence_lengths": list(self.sequence_lengths),
            "head_dims": list(self.head_dims),
            "dtype": "bfloat16",
            "is_causal": True,
            "seed": self.seed,
            "timer": "triton.testing.do_bench_or_cuda_events",
            "warmup_ms": self.warmup_ms,
            "rep_ms": self.rep_ms,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Directory for attention_baseline.csv and metadata.")
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(formal=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--formal", dest="formal", action="store_true", help="Require one RTX 4090 with at least 22 GiB free before running.")
    mode.add_argument(
        "--non-formal",
        dest="formal",
        action="store_false",
        help="Development-only: permit non-4090 or reduced-matrix measurements; outputs are tagged formal=false.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-lengths", default="512,2048,8192")
    parser.add_argument("--head-dims", default="64,128")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    return parser


def _base_row(config: AttentionBenchmarkConfig, *, sequence_length: int | None, head_dim: int | None, phase: str) -> dict[str, object]:
    return {
        "implementation": "eager_explicit_pytorch",
        "batch_size": config.batch_size,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": dtype_name(torch.bfloat16),
        "is_causal": True,
        "phase": phase,
        "seed": config.seed,
        "formal": config.formal,
        "warmup_ms": config.warmup_ms,
        "rep_ms": config.rep_ms,
        "allocator_limit_mib": None,
        "allocator_fraction": None,
        "reason": None,
    }


def _exception_measurement(error: Exception) -> PhaseMeasurement:
    allocated, reserved = current_peak_memory_mib()
    is_oom = is_out_of_memory(error)
    return PhaseMeasurement(
        status="oom" if is_oom else "failed",
        error_kind="oom" if is_oom else error_kind(error),
        latency=None,
        peak_allocated_mib=allocated,
        peak_reserved_mib=reserved,
    )


def _validate(config: AttentionBenchmarkConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if config.warmup_ms <= 0 or config.rep_ms <= 0:
        raise ValueError("--warmup-ms and --rep-ms must be positive.")
    if config.formal:
        if config.batch_size != 1:
            raise ValueError("Formal attention baseline mode requires batch size 1.")
        if config.sequence_lengths != (512, 2048, 8192) or config.head_dims != (64, 128):
            raise ValueError("Formal attention baseline mode requires the fixed 512/2048/8192 × 64/128 matrix.")
        if config.warmup_ms != 100 or config.rep_ms != 300:
            raise ValueError("Formal attention baseline mode requires warmup=100 ms and rep=300 ms.")


def run(config: AttentionBenchmarkConfig, *, output_dir: Path, device_name: str) -> int:
    _validate(config)
    result_path = output_dir / "attention_baseline.csv"
    try:
        runtime = configure_cuda(device_name, formal=config.formal)
    except CudaPreflightError as error:
        unavailable_row = _base_row(config, sequence_length=None, head_dim=None, phase="not_run")
        unavailable_row.update(
            {
                "status": error.status,
                "error_kind": "cuda_preflight",
                "reason": error.public_reason,
            }
        )
        write_csv(result_path, RESULT_FIELDS, [unavailable_row])
        record_preflight_failure(
            output_dir,
            script_name="benchmark_attention.py",
            formal=config.formal,
            configuration=config.as_json(),
            error=error,
        )
        stderr(error.public_reason)
        return 2

    rows: list[dict[str, object]] = []
    measurements: list[PhaseMeasurement] = []
    write_csv(result_path, RESULT_FIELDS, rows)
    for sequence_length in config.sequence_lengths:
        for head_dim in config.head_dims:
            for phase in PHASES:
                cleanup_cuda()
                row = _base_row(config, sequence_length=sequence_length, head_dim=head_dim, phase=phase)
                try:
                    torch.cuda.reset_peak_memory_stats(runtime.device)
                    q, k, v, grad_output = make_attention_inputs(
                        batch_size=config.batch_size,
                        sequence_length=sequence_length,
                        head_dim=head_dim,
                        dtype=torch.bfloat16,
                        device=runtime.device,
                        seed=config.seed,
                    )
                    workload = make_attention_phase(
                        explicit_attention_apply,
                        q=q,
                        k=k,
                        v=v,
                        grad_output=grad_output,
                        is_causal=True,
                        phase=phase,  # type: ignore[arg-type]
                    )
                    measurement = measure_cuda_workload(workload, warmup_ms=config.warmup_ms, rep_ms=config.rep_ms)
                except Exception as error:
                    measurement = _exception_measurement(error)
                finally:
                    # The backward workload closes over Q/K/V and can retain an
                    # autograd graph. Drop those references before clearing the
                    # allocator cache so the next row never overlaps inputs
                    # with the previous row's graph.
                    workload = None
                    grad_output = None
                    v = None
                    k = None
                    q = None
                    cleanup_cuda()
                row.update(measurement_csv_fields(measurement))
                row["allocator_limit_mib"] = 23 * 1024
                row["allocator_fraction"] = runtime.allocator_fraction
                rows.append(row)
                measurements.append(measurement)
                write_csv(result_path, RESULT_FIELDS, rows)

    peak_allocated, peak_reserved = maximum_peak(measurements)
    status = "success" if rows and all(row["status"] == "success" for row in rows) else "incomplete"
    append_run_metadata(
        output_dir,
        script_name="benchmark_attention.py",
        runtime=runtime,
        status=status,
        formal=config.formal,
        configuration=config.as_json(),
    )
    append_memory_observation(
        output_dir,
        script_name="benchmark_attention.py",
        runtime=runtime,
        status=status,
        peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,
        formal=config.formal,
    )
    if status != "success":
        stderr("Attention baseline matrix contains OOM, failed, or allocator-limit rows; no complete benchmark was reported.")
        return 1
    print(f"Wrote {len(rows)} real eager-attention benchmark rows to {result_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = AttentionBenchmarkConfig(
            batch_size=args.batch_size,
            sequence_lengths=parse_positive_ints(args.sequence_lengths, option="--sequence-lengths"),
            head_dims=parse_positive_ints(args.head_dims, option="--head-dims"),
            seed=args.seed,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            formal=bool(args.formal),
        )
        return run(config, output_dir=args.output_dir, device_name=args.device)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
