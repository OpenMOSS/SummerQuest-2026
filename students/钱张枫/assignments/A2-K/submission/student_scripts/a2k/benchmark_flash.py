"""Benchmark equivalent eager, compiled, and student Triton FlashAttention paths.

The core matrix compares all three implementations at the prescribed shapes.
The 16384 boundary keeps eager and the student Triton implementation even when
eager OOMs; it never replaces an OOM row with a smaller input.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

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
        disable_aot_donated_buffers,
        dtype_name,
        error_kind,
        explicit_attention_apply,
        is_out_of_memory,
        load_flash_implementations,
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
        disable_aot_donated_buffers,
        dtype_name,
        error_kind,
        explicit_attention_apply,
        is_out_of_memory,
        load_flash_implementations,
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
    "matrix",
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
    "speedup_vs_eager",
    "query_tile",
    "key_tile",
    "num_warps",
    "num_stages",
    "compile_backend",
    "status",
    "error_kind",
    "reason",
)


@dataclass(frozen=True)
class FlashBenchmarkConfig:
    batch_size: int
    core_sequence_lengths: tuple[int, ...]
    long_sequence_length: int
    head_dims: tuple[int, ...]
    include_compiled_long: bool
    seed: int
    warmup_ms: int
    rep_ms: int
    formal: bool

    def as_json(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "core_sequence_lengths": list(self.core_sequence_lengths),
            "long_sequence_length": self.long_sequence_length,
            "head_dims": list(self.head_dims),
            "dtype": "bfloat16",
            "is_causal": True,
            "phases": list(PHASES),
            "include_compiled_long": self.include_compiled_long,
            "seed": self.seed,
            "warmup_ms": self.warmup_ms,
            "rep_ms": self.rep_ms,
            "compiled_backward_graph_reuse": "aot_donated_buffer_disabled",
            "timer": "triton.testing.do_bench_or_cuda_events",
        }


@dataclass(frozen=True)
class ImplementationSpec:
    name: str
    apply: Callable[[Tensor, Tensor, Tensor, bool], Tensor] | None
    kernel_config: dict[str, int | None]
    compile_backend: str | None
    unavailable_error_kind: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Directory for flash_benchmark.csv and metadata.")
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(formal=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--formal", dest="formal", action="store_true", help="Require a single RTX 4090 with at least 22 GiB free.")
    mode.add_argument(
        "--non-formal",
        dest="formal",
        action="store_false",
        help="Development-only: permit non-4090 or reduced-matrix measurements; outputs are tagged formal=false.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-lengths", default="512,2048,8192", help="Core sequence lengths.")
    parser.add_argument("--long-sequence-length", type=int, default=16384)
    parser.add_argument("--head-dims", default="64,128")
    parser.add_argument("--include-compiled-long", action="store_true", help="Also attempt compiled explicit attention at sequence length 16384.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    return parser


def _validate(config: FlashBenchmarkConfig) -> None:
    if config.batch_size <= 0 or config.long_sequence_length <= 0:
        raise ValueError("Batch size and long sequence length must be positive.")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if config.warmup_ms <= 0 or config.rep_ms <= 0:
        raise ValueError("--warmup-ms and --rep-ms must be positive.")
    if config.formal:
        if config.batch_size != 1:
            raise ValueError("Formal Flash benchmark mode requires batch size 1.")
        if config.core_sequence_lengths != (512, 2048, 8192) or config.long_sequence_length != 16384:
            raise ValueError("Formal Flash benchmark mode requires core 512/2048/8192 and long sequence 16384.")
        if config.head_dims != (64, 128):
            raise ValueError("Formal Flash benchmark mode requires head dimensions 64 and 128.")
        if config.warmup_ms != 100 or config.rep_ms != 300:
            raise ValueError("Formal Flash benchmark mode requires warmup=100 ms and rep=300 ms.")


def _base_row(
    config: FlashBenchmarkConfig,
    *,
    implementation: str,
    matrix: str,
    sequence_length: int | None,
    head_dim: int | None,
    phase: str,
    kernel_config: dict[str, int | None],
    compile_backend: str | None,
) -> dict[str, object]:
    return {
        "implementation": implementation,
        "matrix": matrix,
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
        "speedup_vs_eager": None,
        "query_tile": kernel_config["query_tile"],
        "key_tile": kernel_config["key_tile"],
        "num_warps": kernel_config["num_warps"],
        "num_stages": kernel_config["num_stages"],
        "compile_backend": compile_backend,
        "reason": None,
    }


def _exception_measurement(error: Exception, *, compiled: bool) -> PhaseMeasurement:
    allocated, reserved = current_peak_memory_mib()
    if is_out_of_memory(error):
        return PhaseMeasurement("oom", "oom", None, allocated, reserved)
    return PhaseMeasurement("compile_error" if compiled else "failed", "compile_error" if compiled else error_kind(error), None, allocated, reserved)


def _compile_explicit_attention() -> Callable[[Tensor, Tensor, Tensor, bool], Tensor]:
    """Compile only the same explicit baseline used by the eager comparison."""

    disable_aot_donated_buffers()
    from cs336_systems.a2k.attention import explicit_attention

    def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return explicit_attention(q, k, v, is_causal=True)

    compiled = torch.compile(attention, fullgraph=False)

    def apply(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
        if not is_causal:
            raise ValueError("The A2-K performance matrix is fixed to causal attention.")
        return compiled(q, k, v)

    return apply


def _triton_spec() -> ImplementationSpec:
    try:
        implementations = load_flash_implementations(include_pytorch=False, include_triton=True)
        implementation = implementations[0]
        return ImplementationSpec(
            name=implementation.name,
            apply=implementation.apply,
            kernel_config=implementation.kernel_config,
            compile_backend=None,
        )
    except Exception as error:
        return ImplementationSpec(
            name="triton_flashattention2",
            apply=None,
            kernel_config={"query_tile": None, "key_tile": None, "num_warps": None, "num_stages": None},
            compile_backend=None,
            unavailable_error_kind=error_kind(error),
        )


def _specifications_for_row(*, include_compiled: bool, triton: ImplementationSpec) -> list[ImplementationSpec]:
    specifications = [
        ImplementationSpec(
            name="eager_explicit_pytorch",
            apply=explicit_attention_apply,
            kernel_config={"query_tile": None, "key_tile": None, "num_warps": None, "num_stages": None},
            compile_backend=None,
        )
    ]
    if include_compiled:
        try:
            compiled_apply = _compile_explicit_attention()
            specifications.append(
                ImplementationSpec(
                    name="compiled_explicit_pytorch",
                    apply=compiled_apply,
                    kernel_config={"query_tile": None, "key_tile": None, "num_warps": None, "num_stages": None},
                    compile_backend="inductor_default",
                )
            )
        except Exception as error:
            specifications.append(
                ImplementationSpec(
                    name="compiled_explicit_pytorch",
                    apply=None,
                    kernel_config={"query_tile": None, "key_tile": None, "num_warps": None, "num_stages": None},
                    compile_backend="inductor_default",
                    unavailable_error_kind="compile_error" if error_kind(error) != "oom" else "oom",
                )
            )
    specifications.append(triton)
    return specifications


def _run_row(
    spec: ImplementationSpec,
    *,
    config: FlashBenchmarkConfig,
    runtime_device: torch.device,
    sequence_length: int,
    head_dim: int,
    phase: str,
) -> PhaseMeasurement:
    if spec.apply is None:
        return PhaseMeasurement(
            status="compile_error" if spec.name.startswith("compiled") else "failed",
            error_kind=spec.unavailable_error_kind,
            latency=None,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
        )
    try:
        torch.cuda.reset_peak_memory_stats(runtime_device)
        q, k, v, grad_output = make_attention_inputs(
            batch_size=config.batch_size,
            sequence_length=sequence_length,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=runtime_device,
            seed=config.seed,
        )
        workload = make_attention_phase(
            spec.apply,
            q=q,
            k=k,
            v=v,
            grad_output=grad_output,
            is_causal=True,
            phase=phase,  # type: ignore[arg-type]
        )
        return measure_cuda_workload(workload, warmup_ms=config.warmup_ms, rep_ms=config.rep_ms)
    except Exception as error:
        return _exception_measurement(error, compiled=spec.name.startswith("compiled"))


def _calculate_speedups(rows: list[dict[str, object]]) -> None:
    eager_p50: dict[tuple[object, ...], float] = {}
    for row in rows:
        if row["implementation"] == "eager_explicit_pytorch" and row.get("status") == "success" and isinstance(row.get("p50_ms"), (int, float)):
            key = (row["sequence_length"], row["head_dim"], row["dtype"], row["is_causal"], row["phase"], row["batch_size"])
            eager_p50[key] = float(row["p50_ms"])
    for row in rows:
        key = (row["sequence_length"], row["head_dim"], row["dtype"], row["is_causal"], row["phase"], row["batch_size"])
        own_p50 = row.get("p50_ms")
        if row.get("status") == "success" and isinstance(own_p50, (int, float)) and key in eager_p50 and float(own_p50) > 0:
            row["speedup_vs_eager"] = round(eager_p50[key] / float(own_p50), 6)
        else:
            row["speedup_vs_eager"] = None


def run(config: FlashBenchmarkConfig, *, output_dir: Path, device_name: str) -> int:
    _validate(config)
    result_path = output_dir / "flash_benchmark.csv"
    try:
        runtime = configure_cuda(device_name, formal=config.formal)
    except CudaPreflightError as error:
        row = _base_row(
            config,
            implementation="not_run",
            matrix="not_run",
            sequence_length=None,
            head_dim=None,
            phase="not_run",
            kernel_config={"query_tile": None, "key_tile": None, "num_warps": None, "num_stages": None},
            compile_backend=None,
        )
        row.update({"status": error.status, "error_kind": "cuda_preflight", "reason": error.public_reason})
        write_csv(result_path, RESULT_FIELDS, [row])
        record_preflight_failure(
            output_dir,
            script_name="benchmark_flash.py",
            formal=config.formal,
            configuration=config.as_json(),
            error=error,
        )
        stderr(error.public_reason)
        return 2

    triton = _triton_spec()
    rows: list[dict[str, object]] = []
    measurements: list[PhaseMeasurement] = []
    write_csv(result_path, RESULT_FIELDS, rows)

    matrices = [("core", sequence_length, True) for sequence_length in config.core_sequence_lengths]
    matrices.append(("long_sequence", config.long_sequence_length, config.include_compiled_long))
    for matrix, sequence_length, include_compiled in matrices:
        for head_dim in config.head_dims:
            # Reuse each shape-specialized implementation across all phases.
            # Rebuilding this list inside the phase loop creates independent
            # torch.compile wrappers and duplicate compilation caches.
            specifications = _specifications_for_row(include_compiled=include_compiled, triton=triton)
            for phase in PHASES:
                for spec in specifications:
                    cleanup_cuda()
                    row = _base_row(
                        config,
                        implementation=spec.name,
                        matrix=matrix,
                        sequence_length=sequence_length,
                        head_dim=head_dim,
                        phase=phase,
                        kernel_config=spec.kernel_config,
                        compile_backend=spec.compile_backend,
                    )
                    measurement = _run_row(
                        spec,
                        config=config,
                        runtime_device=runtime.device,
                        sequence_length=sequence_length,
                        head_dim=head_dim,
                        phase=phase,
                    )
                    cleanup_cuda()
                    row.update(measurement_csv_fields(measurement))
                    row["allocator_limit_mib"] = 23 * 1024
                    row["allocator_fraction"] = runtime.allocator_fraction
                    rows.append(row)
                    measurements.append(measurement)
                    write_csv(result_path, RESULT_FIELDS, rows)

    _calculate_speedups(rows)
    write_csv(result_path, RESULT_FIELDS, rows)
    peak_allocated, peak_reserved = maximum_peak(measurements)
    status = "success" if rows and all(row["status"] == "success" for row in rows) else "incomplete"
    append_run_metadata(
        output_dir,
        script_name="benchmark_flash.py",
        runtime=runtime,
        status=status,
        formal=config.formal,
        configuration=config.as_json(),
    )
    append_memory_observation(
        output_dir,
        script_name="benchmark_flash.py",
        runtime=runtime,
        status=status,
        peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,
        formal=config.formal,
    )
    if status != "success":
        stderr("Flash matrix contains OOM, compile-error, failed, or allocator-limit rows; inspect flash_benchmark.csv.")
        return 1
    print(f"Wrote {len(rows)} real FlashAttention benchmark rows to {result_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = FlashBenchmarkConfig(
            batch_size=args.batch_size,
            core_sequence_lengths=parse_positive_ints(args.sequence_lengths, option="--sequence-lengths"),
            long_sequence_length=args.long_sequence_length,
            head_dims=parse_positive_ints(args.head_dims, option="--head-dims"),
            include_compiled_long=bool(args.include_compiled_long),
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
