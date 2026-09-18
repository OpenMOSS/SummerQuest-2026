"""Compare eager and ``torch.compile`` attention plus a real Stanford-small model.

Cold-start wall-clock time is stored separately from steady-state CUDA latency.
The former intentionally includes compiler work; the latter uses the A2-K
warmup/measurement protocol and excludes input/model/optimizer construction.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

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
        make_attention_inputs,
        make_attention_phase,
        maximum_peak,
        measurement_csv_fields,
        measure_cuda_workload,
        parse_attention_shapes,
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
        make_attention_inputs,
        make_attention_phase,
        maximum_peak,
        measurement_csv_fields,
        measure_cuda_workload,
        parse_attention_shapes,
        record_preflight_failure,
        stderr,
        write_csv,
    )


ATTENTION_PHASES: tuple[str, ...] = ("forward", "backward", "forward_backward")
MODEL_PHASES: tuple[str, ...] = ("forward", "forward_backward", "training_step")
SMALL_MODEL_CONFIG: dict[str, int | float | None] = {
    "vocab_size": 10_000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": 10_000.0,
}
RESULT_FIELDS: tuple[str, ...] = (
    "implementation",
    "workload",
    "phase",
    "batch_size",
    "sequence_length",
    "head_dim",
    "num_heads",
    "dtype",
    "is_causal",
    "seed",
    "formal",
    "model_config",
    "compile_backend",
    "cold_start_ms_wall",
    "cold_start_scope",
    "cold_start_status",
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
class CompileBenchmarkConfig:
    attention_shapes: tuple[tuple[int, int], ...]
    batch_size: int
    context_length: int
    seed: int
    warmup_ms: int
    rep_ms: int
    formal: bool
    skip_model: bool

    def as_json(self) -> dict[str, object]:
        return {
            "attention_shapes": [{"sequence_length": sequence_length, "head_dim": head_dim} for sequence_length, head_dim in self.attention_shapes],
            "attention_dtype": "bfloat16",
            "attention_is_causal": True,
            "model": {**SMALL_MODEL_CONFIG, "context_length": self.context_length},
            "model_dtype": "bfloat16_autocast_with_fp32_parameters",
            "batch_size": self.batch_size,
            "seed": self.seed,
            "warmup_ms": self.warmup_ms,
            "rep_ms": self.rep_ms,
            "compiled_backward_graph_reuse": "aot_donated_buffer_disabled",
            "timer": "triton.testing.do_bench_or_cuda_events",
            "skip_model": self.skip_model,
        }


@dataclass(frozen=True)
class CompiledCallable:
    apply: Callable[[Tensor, Tensor, Tensor, bool], Tensor] | None
    error_kind_value: str | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Directory for compile_comparison.csv and metadata.")
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(formal=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--formal", dest="formal", action="store_true", help="Enforce single RTX 4090 / 22 GiB free-memory preflight checks.")
    mode.add_argument(
        "--non-formal",
        dest="formal",
        action="store_false",
        help="Development-only: permit non-4090 or reduced-matrix measurements; outputs are tagged formal=false.",
    )
    parser.add_argument("--attention-shapes", default="512:64,2048:128,8192:128")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--skip-model", action="store_true", help="Development-only: skip the required small-model comparison.")
    return parser


def _validate(config: CompileBenchmarkConfig) -> None:
    if config.batch_size <= 0 or config.context_length <= 0:
        raise ValueError("--batch-size and --context-length must be positive.")
    if config.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if config.warmup_ms <= 0 or config.rep_ms <= 0:
        raise ValueError("--warmup-ms and --rep-ms must be positive.")
    if config.formal:
        if config.batch_size != 1 or config.context_length != 512:
            raise ValueError("Formal compile mode requires batch size 1 and context length 512.")
        if config.attention_shapes != ((512, 64), (2048, 128), (8192, 128)):
            raise ValueError("Formal compile mode requires attention shapes 512:64, 2048:128, and 8192:128.")
        if config.warmup_ms != 100 or config.rep_ms != 300:
            raise ValueError("Formal compile mode requires warmup=100 ms and rep=300 ms.")
        if config.skip_model:
            raise ValueError("Formal compile mode cannot skip the Stanford-small model comparison.")


def _base_row(
    config: CompileBenchmarkConfig,
    *,
    implementation: str,
    workload: str,
    phase: str,
    sequence_length: int | None,
    head_dim: int | None,
    num_heads: int | None,
    model_config: str | None,
) -> dict[str, object]:
    return {
        "implementation": implementation,
        "workload": workload,
        "phase": phase,
        "batch_size": config.batch_size,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "dtype": dtype_name(torch.bfloat16),
        "is_causal": True if workload == "attention" else None,
        "seed": config.seed,
        "formal": config.formal,
        "model_config": model_config,
        "compile_backend": "inductor_default" if implementation == "compiled" else None,
        "cold_start_ms_wall": None,
        "cold_start_scope": None,
        "cold_start_status": "not_applicable" if implementation == "eager" else "not_run",
        "warmup_ms": config.warmup_ms,
        "rep_ms": config.rep_ms,
        "allocator_limit_mib": None,
        "allocator_fraction": None,
        "reason": None,
    }


def _compile_attention() -> CompiledCallable:
    try:
        disable_aot_donated_buffers()
        from cs336_systems.a2k.attention import explicit_attention

        def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            return explicit_attention(q, k, v, is_causal=True)

        compiled = torch.compile(attention, fullgraph=False)

        def apply(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
            if not is_causal:
                raise ValueError("The fixed performance matrix is causal.")
            return compiled(q, k, v)

        return CompiledCallable(apply=apply, error_kind_value=None)
    except Exception as error:
        return CompiledCallable(apply=None, error_kind_value="oom" if is_out_of_memory(error) else "compile_error")


def _cold_start(workload: Callable[[], object]) -> tuple[float | None, str | None]:
    """Measure first execution with wall time, including compiler CPU work."""

    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = workload()
        del result
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0, None
    except Exception as error:
        return None, "oom" if is_out_of_memory(error) else "compile_error"


def _exception_measurement(error: Exception, *, compiled: bool) -> PhaseMeasurement:
    allocated, reserved = current_peak_memory_mib()
    if is_out_of_memory(error):
        return PhaseMeasurement("oom", "oom", None, allocated, reserved)
    return PhaseMeasurement("compile_error" if compiled else "failed", "compile_error" if compiled else error_kind(error), None, allocated, reserved)


def _measure_attention(
    apply: Callable[[Tensor, Tensor, Tensor, bool], Tensor] | None,
    *,
    compiled: bool,
    unavailable_error: str | None,
    config: CompileBenchmarkConfig,
    device: torch.device,
    sequence_length: int,
    head_dim: int,
    phase: str,
) -> tuple[PhaseMeasurement, float | None, str | None]:
    if apply is None:
        return (
            PhaseMeasurement("compile_error" if compiled else "failed", unavailable_error, None, None, None),
            None,
            unavailable_error,
        )
    try:
        torch.cuda.reset_peak_memory_stats(device)
        q, k, v, grad_output = make_attention_inputs(
            batch_size=config.batch_size,
            sequence_length=sequence_length,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device=device,
            seed=config.seed,
        )

        def build_workload() -> Callable[[], object]:
            return make_attention_phase(
                apply,
                q=q,
                k=k,
                v=v,
                grad_output=grad_output,
                is_causal=True,
                phase=phase,  # type: ignore[arg-type]
            )

        cold_ms: float | None = None
        cold_error: str | None = None
        if compiled:
            cold_ms, cold_error = _cold_start(build_workload())
            if cold_error is not None:
                return PhaseMeasurement("oom" if cold_error == "oom" else "compile_error", cold_error, None, *current_peak_memory_mib()), cold_ms, cold_error
        return measure_cuda_workload(build_workload(), warmup_ms=config.warmup_ms, rep_ms=config.rep_ms), cold_ms, cold_error
    except Exception as error:
        return _exception_measurement(error, compiled=compiled), None, "oom" if is_out_of_memory(error) else ("compile_error" if compiled else error_kind(error))


def _build_small_model(*, device: torch.device, context_length: int, seed: int) -> tuple[nn.Module, torch.optim.Optimizer, Tensor, str]:
    """Instantiate the assignment's Stanford-small model outside every timing interval."""

    from cs336_basics.model import BasicsTransformerLM

    torch.manual_seed(seed)
    model_arguments: dict[str, int | float | None] = {**SMALL_MODEL_CONFIG, "context_length": context_length}
    model = BasicsTransformerLM(**model_arguments).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    token_ids = torch.randint(
        0,
        int(model_arguments["vocab_size"]),
        (1, context_length),
        device=device,
        dtype=torch.long,
    )
    model_config = json.dumps(model_arguments, sort_keys=True, separators=(",", ":"))
    return model, optimizer, token_ids, model_config


def _model_workload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: Tensor,
    *,
    phase: str,
) -> Callable[[], object]:
    def forward() -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(token_ids)

    if phase == "forward":
        return forward
    if phase == "forward_backward":

        def forward_backward() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            output = forward()
            loss = output.float().mean()
            loss.backward()
            return loss

        return forward_backward
    if phase == "training_step":

        def training_step() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            output = forward()
            loss = output.float().mean()
            loss.backward()
            optimizer.step()
            return loss

        return training_step
    raise ValueError(f"Unsupported model phase {phase!r}.")


def _measure_model(
    *,
    compiled: bool,
    config: CompileBenchmarkConfig,
    device: torch.device,
    phase: str,
) -> tuple[PhaseMeasurement, float | None, str | None, str | None]:
    try:
        model, optimizer, token_ids, model_config = _build_small_model(device=device, context_length=config.context_length, seed=config.seed)
        measured_model: nn.Module
        if compiled:
            measured_model = torch.compile(model, fullgraph=False)
        else:
            measured_model = model

        def build_workload() -> Callable[[], object]:
            return _model_workload(measured_model, optimizer, token_ids, phase=phase)

        cold_ms: float | None = None
        cold_error: str | None = None
        if compiled:
            cold_ms, cold_error = _cold_start(build_workload())
            if cold_error is not None:
                return (
                    PhaseMeasurement("oom" if cold_error == "oom" else "compile_error", cold_error, None, *current_peak_memory_mib()),
                    cold_ms,
                    cold_error,
                    model_config,
                )
        measurement = measure_cuda_workload(build_workload(), warmup_ms=config.warmup_ms, rep_ms=config.rep_ms)
        return measurement, cold_ms, cold_error, model_config
    except Exception as error:
        return (
            _exception_measurement(error, compiled=compiled),
            None,
            "oom" if is_out_of_memory(error) else ("compile_error" if compiled else error_kind(error)),
            None,
        )


def _unavailable_row(config: CompileBenchmarkConfig, *, status: str, reason: str) -> dict[str, object]:
    row = _base_row(
        config,
        implementation="not_run",
        workload="not_run",
        phase="not_run",
        sequence_length=None,
        head_dim=None,
        num_heads=None,
        model_config=None,
    )
    row.update({"status": status, "error_kind": "cuda_preflight", "reason": reason})
    return row


def run(config: CompileBenchmarkConfig, *, output_dir: Path, device_name: str) -> int:
    _validate(config)
    result_path = output_dir / "compile_comparison.csv"
    try:
        runtime = configure_cuda(device_name, formal=config.formal)
    except CudaPreflightError as error:
        write_csv(result_path, RESULT_FIELDS, [_unavailable_row(config, status=error.status, reason=error.public_reason)])
        record_preflight_failure(
            output_dir,
            script_name="benchmark_compile.py",
            formal=config.formal,
            configuration=config.as_json(),
            error=error,
        )
        stderr(error.public_reason)
        return 2

    rows: list[dict[str, object]] = []
    measurements: list[PhaseMeasurement] = []
    write_csv(result_path, RESULT_FIELDS, rows)

    for sequence_length, head_dim in config.attention_shapes:
        compiled_attention = _compile_attention()
        for implementation, apply, compiled, unavailable_error in (
            ("eager", explicit_attention_apply, False, None),
            ("compiled", compiled_attention.apply, True, compiled_attention.error_kind_value),
        ):
            for phase in ATTENTION_PHASES:
                cleanup_cuda()
                row = _base_row(
                    config,
                    implementation=implementation,
                    workload="attention",
                    phase=phase,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    num_heads=1,
                    model_config=None,
                )
                measurement, cold_ms, cold_error = _measure_attention(
                    apply,
                    compiled=compiled,
                    unavailable_error=unavailable_error,
                    config=config,
                    device=runtime.device,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    phase=phase,
                )
                cleanup_cuda()
                row.update(measurement_csv_fields(measurement))
                row["allocator_limit_mib"] = 23 * 1024
                row["allocator_fraction"] = runtime.allocator_fraction
                row["cold_start_ms_wall"] = round(cold_ms, 6) if cold_ms is not None else None
                row["cold_start_scope"] = "first_phase_invocation_after_setup" if compiled else None
                row["cold_start_status"] = "success" if compiled and cold_error is None else (cold_error or "not_applicable")
                rows.append(row)
                measurements.append(measurement)
                write_csv(result_path, RESULT_FIELDS, rows)

    if not config.skip_model:
        for implementation, compiled in (("eager", False), ("compiled", True)):
            for phase in MODEL_PHASES:
                cleanup_cuda()
                row = _base_row(
                    config,
                    implementation=implementation,
                    workload="stanford_small_model",
                    phase=phase,
                    sequence_length=config.context_length,
                    head_dim=64,
                    num_heads=12,
                    model_config=json.dumps({**SMALL_MODEL_CONFIG, "context_length": config.context_length}, sort_keys=True, separators=(",", ":")),
                )
                measurement, cold_ms, cold_error, model_config = _measure_model(
                    compiled=compiled,
                    config=config,
                    device=runtime.device,
                    phase=phase,
                )
                cleanup_cuda()
                row.update(measurement_csv_fields(measurement))
                row["model_config"] = model_config
                row["allocator_limit_mib"] = 23 * 1024
                row["allocator_fraction"] = runtime.allocator_fraction
                row["cold_start_ms_wall"] = round(cold_ms, 6) if cold_ms is not None else None
                row["cold_start_scope"] = "first_phase_invocation_after_setup" if compiled else None
                row["cold_start_status"] = "success" if compiled and cold_error is None else (cold_error or "not_applicable")
                rows.append(row)
                measurements.append(measurement)
                write_csv(result_path, RESULT_FIELDS, rows)

    peak_allocated, peak_reserved = maximum_peak(measurements)
    status = "success" if rows and all(row["status"] == "success" for row in rows) else "incomplete"
    append_run_metadata(
        output_dir,
        script_name="benchmark_compile.py",
        runtime=runtime,
        status=status,
        formal=config.formal,
        configuration=config.as_json(),
    )
    append_memory_observation(
        output_dir,
        script_name="benchmark_compile.py",
        runtime=runtime,
        status=status,
        peak_allocated_mib=peak_allocated,
        peak_reserved_mib=peak_reserved,
        formal=config.formal,
    )
    if status != "success":
        stderr("Compile comparison contains OOM, compile-error, failed, or allocator-limit rows; inspect compile_comparison.csv.")
        return 1
    print(f"Wrote {len(rows)} real eager/compiled comparison rows to {result_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = CompileBenchmarkConfig(
            attention_shapes=parse_attention_shapes(args.attention_shapes),
            batch_size=args.batch_size,
            context_length=args.context_length,
            seed=args.seed,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
            formal=bool(args.formal),
            skip_model=bool(args.skip_model),
        )
        return run(config, output_dir=args.output_dir, device_name=args.device)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
