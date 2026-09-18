"""Run the A2-K extended FlashAttention correctness matrix on a real CUDA GPU.

Example:

    python student_scripts/a2k/correctness.py --formal --output-dir local_results/a2k

The script never treats a skipped CUDA path as a pass.  When CUDA is absent it
writes an explicit ``unavailable`` result and exits non-zero.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

try:  # Support both `python -m ...` and direct execution from the repository root.
    from .common import (
        CudaPreflightError,
        ALLOCATOR_LIMIT_MIB,
        CudaRuntime,
        FlashImplementation,
        append_memory_observation,
        append_run_metadata,
        cleanup_cuda,
        configure_cuda,
        default_output_dir,
        dtype_from_name,
        dtype_name,
        error_kind,
        explicit_attention_with_lse,
        is_out_of_memory,
        load_flash_implementations,
        make_attention_inputs,
        max_error,
        record_preflight_failure,
        set_tf32,
        stderr,
        write_json,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from common import (  # type: ignore[no-redef]
        CudaPreflightError,
        ALLOCATOR_LIMIT_MIB,
        CudaRuntime,
        FlashImplementation,
        append_memory_observation,
        append_run_metadata,
        cleanup_cuda,
        configure_cuda,
        default_output_dir,
        dtype_from_name,
        dtype_name,
        error_kind,
        explicit_attention_with_lse,
        is_out_of_memory,
        load_flash_implementations,
        make_attention_inputs,
        max_error,
        record_preflight_failure,
        set_tf32,
        stderr,
        write_json,
    )


@dataclass(frozen=True)
class CorrectnessConfig:
    """The recorded experimental configuration, excluding local output paths."""

    batch_size: int
    sequence_length: int
    seeds: tuple[int, ...]
    head_dims: tuple[int, ...]
    dtype_names: tuple[str, ...]
    rtol: float
    atol: float
    formal: bool

    def as_json(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "seeds": list(self.seeds),
            "head_dims": list(self.head_dims),
            "dtypes": list(self.dtype_names),
            "rtol": self.rtol,
            "atol": self.atol,
            "timer": "correctness_comparison",
        }


def _parse_nonnegative_ints(value: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must be a comma-separated list of non-negative integers.") from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError(f"{option} must contain at least one non-negative integer.")
    return values


def _parse_positive_ints(value: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option} must be a comma-separated list of positive integers.") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{option} must contain at least one positive integer.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Directory for correctness.json and metadata.")
    parser.add_argument("--device", default="cuda:0", help="Single CUDA device to measure.")
    parser.set_defaults(formal=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--formal", dest="formal", action="store_true", help="Enforce the RTX 4090 / 22 GiB free-memory preflight checks.")
    mode.add_argument(
        "--non-formal",
        dest="formal",
        action="store_false",
        help="Development-only: permit non-4090 or reduced-matrix measurements; outputs are tagged formal=false.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds; the default has the required three seeds.")
    parser.add_argument("--head-dims", default="32,64,128", help="Comma-separated head dimensions.")
    parser.add_argument("--dtypes", default="fp32", help="Comma-separated dtypes, for example fp32,bf16.")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser


def _metric(actual: Tensor, reference: Tensor, *, rtol: float, atol: float) -> dict[str, object]:
    errors = max_error(actual, reference)
    passed = bool(torch.allclose(actual, reference, rtol=rtol, atol=atol, equal_nan=False))
    return {
        **errors,
        "rtol": rtol,
        "atol": atol,
        "status": "pass" if passed else "fail",
    }


def _extract_lse(output: Tensor, *, batch_size: int, sequence_length: int) -> Tensor:
    grad_fn = output.grad_fn
    if grad_fn is None:
        raise RuntimeError("FlashAttention output has no grad_fn, so its saved LSE cannot be validated.")
    saved_tensors = getattr(grad_fn, "saved_tensors", ())
    matches = [tensor for tensor in saved_tensors if tuple(tensor.shape) == (batch_size, sequence_length)]
    if len(matches) != 1:
        raise RuntimeError("FlashAttention must save exactly one [batch, sequence] LSE tensor.")
    return matches[0].detach().clone()


def _run_case(
    implementation: FlashImplementation,
    *,
    runtime: CudaRuntime,
    config: CorrectnessConfig,
    seed: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
) -> tuple[dict[str, object], float | None, float | None]:
    """Compare O/L/dQ/dK/dV for one real implementation and input configuration."""

    record: dict[str, object] = {
        "implementation": implementation.name,
        "seed": seed,
        "shape": [config.batch_size, config.sequence_length, head_dim],
        "dtype": dtype_name(dtype),
        "is_causal": is_causal,
        "tf32_enabled": False,
        "metrics": {},
        "status": "failed",
        "error_kind": None,
    }
    peak_allocated: float | None = None
    peak_reserved: float | None = None
    try:
        torch.cuda.reset_peak_memory_stats(runtime.device)
        q_base, k_base, v_base, grad_output = make_attention_inputs(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            head_dim=head_dim,
            dtype=dtype,
            device=runtime.device,
            seed=seed,
        )
        torch.cuda.reset_peak_memory_stats(runtime.device)

        q_reference = q_base.detach().clone().requires_grad_(True)
        k_reference = k_base.detach().clone().requires_grad_(True)
        v_reference = v_base.detach().clone().requires_grad_(True)
        reference_output, reference_lse = explicit_attention_with_lse(q_reference, k_reference, v_reference, is_causal)
        reference_gradients = torch.autograd.grad(
            reference_output,
            (q_reference, k_reference, v_reference),
            grad_outputs=grad_output,
            allow_unused=False,
        )

        q_actual = q_base.detach().clone().requires_grad_(True)
        k_actual = k_base.detach().clone().requires_grad_(True)
        v_actual = v_base.detach().clone().requires_grad_(True)
        actual_output = implementation.apply(q_actual, k_actual, v_actual, is_causal)
        actual_lse = _extract_lse(actual_output, batch_size=config.batch_size, sequence_length=config.sequence_length)
        actual_gradients = torch.autograd.grad(
            actual_output,
            (q_actual, k_actual, v_actual),
            grad_outputs=grad_output,
            allow_unused=False,
        )
        torch.cuda.synchronize(runtime.device)
        peak_allocated = torch.cuda.max_memory_allocated(runtime.device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(runtime.device) / (1024**2)

        metrics = {
            "output": _metric(actual_output.detach(), reference_output.detach(), rtol=config.rtol, atol=config.atol),
            "lse": _metric(actual_lse, reference_lse.detach(), rtol=config.rtol, atol=config.atol),
            "dq": _metric(actual_gradients[0], reference_gradients[0], rtol=config.rtol, atol=config.atol),
            "dk": _metric(actual_gradients[1], reference_gradients[1], rtol=config.rtol, atol=config.atol),
            "dv": _metric(actual_gradients[2], reference_gradients[2], rtol=config.rtol, atol=config.atol),
        }
        record["metrics"] = metrics
        passed = all(str(metric["status"]) == "pass" for metric in metrics.values())
        record["status"] = "pass" if passed else "fail"
    except Exception as error:
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated(runtime.device) / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved(runtime.device) / (1024**2)
        record["status"] = "oom" if is_out_of_memory(error) else "failed"
        record["error_kind"] = "oom" if is_out_of_memory(error) else error_kind(error)
    finally:
        cleanup_cuda()
    record["peak_allocated_mib"] = round(peak_allocated, 3) if peak_allocated is not None else None
    record["peak_reserved_mib"] = round(peak_reserved, 3) if peak_reserved is not None else None
    return record, peak_allocated, peak_reserved


def _failure_payload(config: CorrectnessConfig, *, status: str, reason: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "configuration": config.as_json(),
        "records": [],
    }


def run(config: CorrectnessConfig, *, output_dir: Path, device_name: str) -> int:
    result_path = output_dir / "correctness.json"
    if config.batch_size <= 0 or config.sequence_length <= 0:
        raise ValueError("--batch-size and --sequence-length must be positive.")
    if config.rtol < 0 or config.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative.")
    dtypes = tuple(dtype_from_name(name) for name in config.dtype_names)
    if config.formal:
        if len(config.seeds) < 3 or not {32, 64, 128}.issubset(config.head_dims):
            raise ValueError("Formal correctness mode requires at least three seeds and head dimensions 32, 64, and 128.")
        if torch.float32 not in dtypes:
            raise ValueError("Formal correctness mode requires an FP32 configuration with TF32 disabled.")

    try:
        runtime = configure_cuda(device_name, formal=config.formal)
    except CudaPreflightError as error:
        write_json(result_path, _failure_payload(config, status=error.status, reason=error.public_reason))
        record_preflight_failure(
            output_dir,
            script_name="correctness.py",
            formal=config.formal,
            configuration=config.as_json(),
            error=error,
        )
        stderr(error.public_reason)
        return 2

    # At least one required FP32 case must run without TF32.  Keeping it off for
    # every correctness case makes the recorded setting unambiguous.
    set_tf32(False)
    try:
        implementations = load_flash_implementations()
    except Exception as error:
        status = "failed"
        write_json(result_path, _failure_payload(config, status=status, reason=f"Flash adapter unavailable: {error_kind(error)}"))
        append_run_metadata(
            output_dir,
            script_name="correctness.py",
            runtime=runtime,
            status=status,
            formal=config.formal,
            configuration=config.as_json(),
            reason=f"Flash adapter unavailable: {error_kind(error)}",
        )
        append_memory_observation(
            output_dir,
            script_name="correctness.py",
            runtime=runtime,
            status=status,
            peak_allocated_mib=None,
            peak_reserved_mib=None,
            formal=config.formal,
        )
        stderr("Unable to resolve the FlashAttention adapter; no correctness pass was reported.")
        return 1

    records: list[dict[str, object]] = []
    peaks_allocated: list[float] = []
    peaks_reserved: list[float] = []
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "configuration": config.as_json(),
        "records": records,
    }
    write_json(result_path, payload)

    for implementation in implementations:
        for seed in config.seeds:
            for head_dim in config.head_dims:
                for dtype in dtypes:
                    for is_causal in (False, True):
                        record, peak_allocated, peak_reserved = _run_case(
                            implementation,
                            runtime=runtime,
                            config=config,
                            seed=seed,
                            head_dim=head_dim,
                            dtype=dtype,
                            is_causal=is_causal,
                        )
                        records.append(record)
                        if peak_allocated is not None:
                            peaks_allocated.append(peak_allocated)
                        if peak_reserved is not None:
                            peaks_reserved.append(peak_reserved)
                        write_json(result_path, payload)

    all_metrics_passed = bool(records) and all(record["status"] == "pass" for record in records)
    all_peaks_within_guard = bool(records) and all(
        isinstance(record.get("peak_reserved_mib"), (float, int))
        and float(record["peak_reserved_mib"]) <= ALLOCATOR_LIMIT_MIB
        for record in records
    )
    successful = all_metrics_passed and (not config.formal or all_peaks_within_guard)
    final_status = "success" if successful else "failed"
    payload["status"] = final_status
    write_json(result_path, payload)
    append_run_metadata(
        output_dir,
        script_name="correctness.py",
        runtime=runtime,
        status=final_status,
        formal=config.formal,
        configuration=config.as_json(),
    )
    append_memory_observation(
        output_dir,
        script_name="correctness.py",
        runtime=runtime,
        status=final_status,
        peak_allocated_mib=max(peaks_allocated) if peaks_allocated else None,
        peak_reserved_mib=max(peaks_reserved) if peaks_reserved else None,
        formal=config.formal,
    )
    if not successful:
        stderr("Correctness matrix finished with failures or OOM rows; inspect correctness.json before reporting results.")
        return 1
    print(f"Wrote {len(records)} real correctness records to {result_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seeds = _parse_nonnegative_ints(args.seeds, option="--seeds")
        head_dims = _parse_positive_ints(args.head_dims, option="--head-dims")
        dtype_names = tuple(part.strip() for part in args.dtypes.split(",") if part.strip())
        if not dtype_names:
            raise ValueError("--dtypes must not be empty.")
        config = CorrectnessConfig(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            seeds=seeds,
            head_dims=head_dims,
            dtype_names=dtype_names,
            rtol=args.rtol,
            atol=args.atol,
            formal=bool(args.formal),
        )
        return run(config, output_dir=args.output_dir, device_name=args.device)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
