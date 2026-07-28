"""Extended output, LSE, and gradient correctness matrix for A2-K."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import torch

from .common import (
    configure_formal_run,
    public_run_record,
    upsert_json_record,
    write_json,
)

from cs336_systems.a2k import FlashAttentionPyTorch, FlashAttentionTriton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--sequence-length", type=int, default=64)
    return parser.parse_args()


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if is_causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=q.device)
        scores = scores.masked_fill(
            query_positions[:, None] < key_positions[None, :],
            float("-inf"),
        )
    return (
        torch.matmul(torch.softmax(scores, dim=-1), v),
        torch.logsumexp(scores, dim=-1),
    )


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    actual_float = actual.float()
    expected_float = expected.float()
    absolute = (actual_float - expected_float).abs()
    relative = absolute / expected_float.abs().clamp_min(1e-6)
    threshold = atol + rtol * expected_float.abs()
    return {
        "max_abs_error": float(absolute.max().item()),
        "max_rel_error": float(relative.max().item()),
        "pass": bool(torch.all(absolute <= threshold).item()),
    }


def run_case(
    implementation_name: str,
    implementation: type[torch.autograd.Function],
    *,
    seed: int,
    sequence_length: int,
    head_dim: int,
    is_causal: bool,
    dtype: torch.dtype,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    shape = (2, sequence_length, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    grad_output = torch.randn(shape, device="cuda", dtype=dtype)

    q_reference = q.detach().float().requires_grad_()
    k_reference = k.detach().float().requires_grad_()
    v_reference = v.detach().float().requires_grad_()
    output_reference, lse_reference = reference_attention(
        q_reference,
        k_reference,
        v_reference,
        is_causal,
    )
    output_reference.backward(grad_output.float())

    q_actual = q.detach().clone().requires_grad_()
    k_actual = k.detach().clone().requires_grad_()
    v_actual = v.detach().clone().requires_grad_()
    output_actual = implementation.apply(
        q_actual,
        k_actual,
        v_actual,
        is_causal,
    )
    saved_lse = [
        tensor
        for tensor in output_actual.grad_fn.saved_tensors
        if tensor.shape == lse_reference.shape
    ]
    if len(saved_lse) != 1:
        raise AssertionError(
            "expected exactly one saved LSE tensor with shape [batch, queries]"
        )
    output_actual.backward(grad_output)

    if dtype == torch.float32:
        atol, rtol = 2e-3, 2e-3
    else:
        atol, rtol = 5e-2, 5e-2
    quantities = {
        "output": error_metrics(
            output_actual,
            output_reference,
            atol=atol,
            rtol=rtol,
        ),
        "lse": error_metrics(
            saved_lse[0],
            lse_reference,
            atol=atol,
            rtol=rtol,
        ),
        "grad_q": error_metrics(
            q_actual.grad,
            q_reference.grad,
            atol=atol,
            rtol=rtol,
        ),
        "grad_k": error_metrics(
            k_actual.grad,
            k_reference.grad,
            atol=atol,
            rtol=rtol,
        ),
        "grad_v": error_metrics(
            v_actual.grad,
            v_reference.grad,
            atol=atol,
            rtol=rtol,
        ),
    }
    passed = all(bool(metrics["pass"]) for metrics in quantities.values())
    return {
        "implementation": implementation_name,
        "seed": seed,
        "batch_size": shape[0],
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "causal": is_causal,
        "dtype": str(dtype).removeprefix("torch."),
        "atol": atol,
        "rtol": rtol,
        "quantities": quantities,
        "status": "pass" if passed else "fail",
    }


def main() -> int:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    run = configure_formal_run(seed=seeds[0], tf32_enabled=False)
    implementations = (
        ("pytorch_tiled", FlashAttentionPyTorch),
        ("triton", FlashAttentionTriton),
    )
    cases: list[dict[str, Any]] = []

    for name, implementation in implementations:
        for seed in seeds:
            for head_dim in (32, 64, 128):
                for is_causal in (False, True):
                    case = run_case(
                        name,
                        implementation,
                        seed=seed,
                        sequence_length=args.sequence_length,
                        head_dim=head_dim,
                        is_causal=is_causal,
                        dtype=torch.bfloat16,
                    )
                    cases.append(case)
                    torch.cuda.empty_cache()
        cases.append(
            run_case(
                name,
                implementation,
                seed=seeds[0],
                sequence_length=args.sequence_length,
                head_dim=64,
                is_causal=True,
                dtype=torch.float32,
            )
        )
        torch.cuda.empty_cache()

    passed = sum(case["status"] == "pass" for case in cases)
    payload = {
        "schema_version": 1,
        "assignment": "A2-K",
        "summary": {
            "total": len(cases),
            "passed": passed,
            "failed": len(cases) - passed,
            "skipped": 0,
        },
        "cases": cases,
    }
    write_json(args.output, payload)
    record = public_run_record(
        run=run,
        experiment="extended_correctness",
        command=(
            "python -m student_scripts.a2k.correctness "
            "--seeds 0,1,2 --sequence-length 64"
        ),
        timer="not applicable; correctness only",
        warmup={"kind": "none"},
        measurement={
            "implementations": [name for name, _ in implementations],
            "seeds": seeds,
            "head_dims": [32, 64, 128],
            "causal": [False, True],
            "dtypes": ["bfloat16", "float32"],
        },
        extra={"passed": passed, "total": len(cases)},
    )
    upsert_json_record(
        args.metadata,
        record,
        key_fields=("experiment",),
    )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
