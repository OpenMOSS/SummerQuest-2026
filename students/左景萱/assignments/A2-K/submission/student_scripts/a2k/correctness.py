#!/usr/bin/env python3
"""Extended A2-K correctness matrix for tiled PyTorch and student Triton.

The authoritative matrix covers BF16 across three seeds, three head dimensions,
and both causal modes, plus a TF32-disabled FP32 sentinel.  Every implementation
is compared against an explicit FP32 attention reference for O, L, dQ, dK, and
dV; no fused attention API is used.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import math
from pathlib import Path
from typing import Any

from student_scripts.a2k.runtime import (
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


SCHEMA_VERSION = "cs336.a2k.correctness.v1"
SEEDS = (17, 42, 336)
HEAD_DIMS = (32, 64, 128)
CAUSAL_MODES = (False, True)
SEQUENCE_LENGTH = 128
BATCH_SIZE = 1
TENSOR_NAMES = ("O", "L", "dQ", "dK", "dV")
TOLERANCES: dict[str, dict[str, float]] = {
    "bfloat16": {"atol": 4e-2, "rtol": 5e-2},
    "float32": {"atol": 2e-4, "rtol": 2e-4},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the A2-K extended O/L/dQ/dK/dV correctness matrix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one tiny CPU PyTorch-tiled check; never count it as GPU evidence",
    )
    parser.add_argument(
        "--development-cuda",
        action="store_true",
        help="run exact cases with a 23 GiB cap on non-standard hardware; never formal evidence",
    )
    return parser


def correctness_cases(*, dry_run: bool) -> list[dict[str, Any]]:
    if dry_run:
        return [
            {
                "seed": SEEDS[0],
                "head_dim": HEAD_DIMS[0],
                "causal": False,
                "dtype": "float32",
                "sequence_length": 16,
            }
        ]
    cases = [
        {
            "seed": seed,
            "head_dim": head_dim,
            "causal": causal,
            "dtype": "bfloat16",
            "sequence_length": SEQUENCE_LENGTH,
        }
        for seed in SEEDS
        for head_dim in HEAD_DIMS
        for causal in CAUSAL_MODES
    ]
    # The sentinel explicitly proves that FP32 was run with TF32 disabled. The
    # full combinatorial coverage is already supplied by the BF16 cases above.
    cases.append(
        {
            "seed": SEEDS[0],
            "head_dim": 128,
            "causal": True,
            "dtype": "float32",
            "sequence_length": SEQUENCE_LENGTH,
        }
    )
    return cases


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _make_inputs(
    *,
    seed: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (BATCH_SIZE, sequence_length, head_dim)
    cpu_values = [torch.randn(shape, generator=generator, dtype=torch.float32) for _ in range(4)]
    query, key, value, grad_output = (tensor.to(device=device, dtype=dtype) for tensor in cpu_values)
    return query, key, value, grad_output


def _explicit_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    causal: bool,
) -> dict[str, torch.Tensor]:
    q_ref = query.detach().float().requires_grad_(True)
    k_ref = key.detach().float().requires_grad_(True)
    v_ref = value.detach().float().requires_grad_(True)
    scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if causal:
        n_queries, n_keys = scores.shape[-2:]
        query_positions = torch.arange(n_queries, device=scores.device).unsqueeze(-1)
        key_positions = torch.arange(n_keys, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill(key_positions > query_positions, -torch.inf)
    logsumexp = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v_ref)
    output.backward(grad_output.float())
    synchronize(query.device)
    return {
        "O": output.detach(),
        "L": logsumexp.detach(),
        "dQ": q_ref.grad.detach(),
        "dK": k_ref.grad.detach(),
        "dV": v_ref.grad.detach(),
    }


def _extract_lse(output: torch.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    grad_fn = output.grad_fn
    if grad_fn is None:
        raise RuntimeError("attention output has no autograd context")
    candidates = [tensor for tensor in grad_fn.saved_tensors if tuple(tensor.shape) == expected_shape]
    if len(candidates) != 1:
        raise RuntimeError("attention must save exactly one [batch, n_queries] log-sum-exp tensor")
    return candidates[0]


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    actual_fp32 = actual.detach().float()
    expected_fp32 = expected.detach().float()
    difference = (actual_fp32 - expected_fp32).abs()
    denominator = expected_fp32.abs().clamp_min(1e-7)
    close = difference <= atol + rtol * expected_fp32.abs()
    return {
        "max_absolute_error": float(difference.max().item()),
        "max_relative_error": float((difference / denominator).max().item()),
        "atol": atol,
        "rtol": rtol,
        "passed": bool(close.all().item()),
        "element_count": actual.numel(),
    }


def _run_implementation(
    *,
    operation: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    reference: dict[str, torch.Tensor],
    causal: bool,
    dtype_name: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    q_impl = query.detach().clone().requires_grad_(True)
    k_impl = key.detach().clone().requires_grad_(True)
    v_impl = value.detach().clone().requires_grad_(True)
    reset_peak_memory(device)
    try:
        output = operation(q_impl, k_impl, v_impl, causal)
        logsumexp = _extract_lse(output, (BATCH_SIZE, query.shape[-2]))
        output.backward(grad_output)
        synchronize(device)
        actual = {
            "O": output.detach(),
            "L": logsumexp.detach(),
            "dQ": q_impl.grad.detach(),
            "dK": k_impl.grad.detach(),
            "dV": v_impl.grad.detach(),
        }
        tolerance = TOLERANCES[dtype_name]
        metrics = {name: _error_metrics(actual[name], reference[name], **tolerance) for name in TENSOR_NAMES}
        result = {
            "status": "pass" if all(metric["passed"] for metric in metrics.values()) else "fail",
            "metrics": metrics,
            "memory": peak_memory_mib(device),
        }
        cpu_actual = {name: tensor.float().cpu() for name, tensor in actual.items()}
        return result, cpu_actual
    except Exception as exc:
        result = {
            "status": "oom" if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower() else "error",
            "error": public_error(exc),
            "memory": peak_memory_mib(device),
        }
        return result, None
    finally:
        q_impl = k_impl = v_impl = None
        release_memory(device)


def _cross_metrics(
    left: dict[str, torch.Tensor] | None,
    right: dict[str, torch.Tensor] | None,
    *,
    dtype_name: str,
) -> dict[str, Any] | None:
    if left is None or right is None:
        return None
    tolerance = TOLERANCES[dtype_name]
    metrics = {name: _error_metrics(right[name], left[name], **tolerance) for name in TENSOR_NAMES}
    return {
        "status": "pass" if all(metric["passed"] for metric in metrics.values()) else "fail",
        "metrics": metrics,
    }


def _case_id(case: dict[str, Any]) -> str:
    causal = "causal" if case["causal"] else "noncausal"
    return f"seed{case['seed']}-n{case['sequence_length']}-d{case['head_dim']}-{case['dtype']}-{causal}"


def run(*, dry_run: bool, development_cuda: bool = False) -> tuple[dict[str, Any], int]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "authoritative": False,
        "reference": "explicit FP32 QK^T / scale / mask / softmax / PV",
        "tf32_required": "ieee",
        "cases": [],
    }
    guard = None
    try:
        guard = prepare_runtime(
            dry_run=dry_run,
            tf32_enabled=False,
            development_cuda=development_cuda,
        )
        payload["runtime"] = guard.metadata
        payload["authoritative"] = guard.authoritative
        from cs336_systems.a2k.attention import FlashAttentionPyTorch, FlashAttentionTriton

        implementations: list[tuple[str, Callable[..., torch.Tensor] | None]] = [
            ("pytorch_tiled", lambda q, k, v, causal: FlashAttentionPyTorch.apply(q, k, v, causal)),
            (
                "triton",
                None if dry_run else (lambda q, k, v, causal: FlashAttentionTriton.apply(q, k, v, causal)),
            ),
        ]
        for case in correctness_cases(dry_run=dry_run):
            dtype = _dtype(case["dtype"])
            query, key, value, grad_output = _make_inputs(
                seed=case["seed"],
                sequence_length=case["sequence_length"],
                head_dim=case["head_dim"],
                dtype=dtype,
                device=guard.device,
            )
            reference = _explicit_reference(query, key, value, grad_output, causal=case["causal"])
            case_record: dict[str, Any] = {
                "case_id": _case_id(case),
                "batch_size": BATCH_SIZE,
                **case,
                "reference_compute_dtype": "float32",
                "tf32_policy": guard.metadata["tf32_policy"],
                "implementations": {},
            }
            observed: dict[str, dict[str, torch.Tensor] | None] = {}
            for name, operation in implementations:
                if operation is None:
                    case_record["implementations"][name] = {
                        "status": "skipped_non_authoritative",
                        "reason": "student Triton requires real CUDA and was not executed by CPU dry-run",
                    }
                    observed[name] = None
                    continue
                result, values = _run_implementation(
                    operation=operation,
                    query=query,
                    key=key,
                    value=value,
                    grad_output=grad_output,
                    reference=reference,
                    causal=case["causal"],
                    dtype_name=case["dtype"],
                    device=guard.device,
                )
                case_record["implementations"][name] = result
                observed[name] = values
            case_record["pytorch_tiled_vs_triton"] = _cross_metrics(
                observed.get("pytorch_tiled"),
                observed.get("triton"),
                dtype_name=case["dtype"],
            )
            statuses = [result["status"] for result in case_record["implementations"].values()]
            expected_statuses = [status for status in statuses if status != "skipped_non_authoritative"]
            case_record["status"] = "pass" if expected_statuses and all(status == "pass" for status in expected_statuses) else "fail"
            payload["cases"].append(case_record)
            query = key = value = grad_output = None
            reference = {}
            release_memory(guard.device)

        checks = [result["status"] for case_record in payload["cases"] for result in case_record["implementations"].values() if result["status"] != "skipped_non_authoritative"]
        payload["summary"] = {
            "case_count": len(payload["cases"]),
            "implementation_check_count": len(checks),
            "passed": checks.count("pass"),
            "failed": checks.count("fail"),
            "errors": checks.count("error"),
            "oom": checks.count("oom"),
            "skipped": sum(result["status"] == "skipped_non_authoritative" for case_record in payload["cases"] for result in case_record["implementations"].values()),
            "bf16_case_count": sum(case["dtype"] == "bfloat16" for case in payload["cases"]),
            "fp32_tf32_disabled_case_count": sum(case["dtype"] == "float32" and case["tf32_policy"] == "ieee" for case in payload["cases"]),
        }
        all_passed = bool(checks) and all(status == "pass" for status in checks)
        payload["status"] = "dry_run_ok" if dry_run and all_passed else ("development_pass" if development_cuda and all_passed else ("pass" if all_passed else "fail"))
        if dry_run:
            payload["non_authoritative_reason"] = guard.metadata["non_authoritative_reason"]
        elif development_cuda:
            payload["non_authoritative_reason"] = guard.metadata["non_authoritative_reason"]
        assert_public_payload(payload)
        return payload, 0 if all_passed else 1
    except Exception as exc:
        payload["status"] = "invalid_environment" if isinstance(exc, RuntimeValidationError) else "error"
        payload["error"] = public_error(exc)
        if guard is not None:
            payload.setdefault("runtime", guard.metadata)
            release_memory(guard.device)
        assert_public_payload(payload)
        return payload, 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.development_cuda:
        parser.error("--dry-run and --development-cuda are mutually exclusive")
    payload, exit_code = run(dry_run=args.dry_run, development_cuda=args.development_cuda)
    atomic_write_json(args.output, payload)
    print(f"correctness: {payload['status']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
