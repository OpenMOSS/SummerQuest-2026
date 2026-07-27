"""Run the extended A2-K FlashAttention correctness matrix."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import torch

from cs336_systems.a2k.attention import FlashAttentionPyTorch, FlashAttentionTriton
from student_scripts.a2k.common import (
    configure_cuda_environment,
    environment_metadata,
    write_json,
)

SEEDS = (0, 1, 2)
HEAD_DIMS = (32, 64, 128)
DTYPES = (torch.float32, torch.bfloat16)
CAUSAL_VALUES = (False, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--sequence-length", type=int, default=128)
    return parser.parse_args()


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if is_causal:
        sequence_length = q.shape[-2]
        indices = torch.arange(sequence_length, device=q.device)
        scores = scores.masked_fill(indices[:, None] < indices[None, :], -1.0e6)
    return torch.matmul(torch.softmax(scores, dim=-1), v), torch.logsumexp(scores.float(), dim=-1)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1.0e-6)
    return difference.max().item(), relative.max().item()


def saved_logsumexp(output: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    matches = [
        tensor
        for tensor in output.grad_fn.saved_tensors
        if tensor.shape == (batch_size, sequence_length)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one saved LSE tensor, found {len(matches)}")
    return matches[0].detach().clone()


def run_case(
    implementation_name: str,
    implementation: type[torch.autograd.Function],
    *,
    seed: int,
    head_dim: int,
    sequence_length: int,
    dtype: torch.dtype,
    is_causal: bool,
) -> dict[str, Any]:
    torch.backends.cuda.matmul.fp32_precision = "ieee" if dtype == torch.float32 else "tf32"
    torch.manual_seed(seed)
    batch_size = 1
    shape = (batch_size, sequence_length, head_dim)
    source_q = torch.randn(shape, device="cuda", dtype=dtype)
    source_k = torch.randn(shape, device="cuda", dtype=dtype)
    source_v = torch.randn(shape, device="cuda", dtype=dtype)
    grad_output = torch.randn(shape, device="cuda", dtype=dtype)

    q_ref = source_q.detach().clone().requires_grad_(True)
    k_ref = source_k.detach().clone().requires_grad_(True)
    v_ref = source_v.detach().clone().requires_grad_(True)
    output_ref, lse_ref = reference_attention(q_ref, k_ref, v_ref, is_causal)
    output_ref.backward(grad_output)

    q = source_q.detach().clone().requires_grad_(True)
    k = source_k.detach().clone().requires_grad_(True)
    v = source_v.detach().clone().requires_grad_(True)
    output = implementation.apply(q, k, v, is_causal)
    lse = saved_logsumexp(output, batch_size, sequence_length)
    output.backward(grad_output)

    tensors = {
        "output": (output, output_ref),
        "logsumexp": (lse, lse_ref),
        "dQ": (q.grad, q_ref.grad),
        "dK": (k.grad, k_ref.grad),
        "dV": (v.grad, v_ref.grad),
    }
    atol = 1.0e-2 if dtype == torch.float32 else 3.0e-2
    rtol = atol
    row: dict[str, Any] = {
        "implementation": implementation_name,
        "seed": seed,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "is_causal": is_causal,
        "tf32_matmul": torch.backends.cuda.matmul.fp32_precision != "ieee",
        "atol": atol,
        "rtol": rtol,
    }
    passed = True
    for name, (actual, expected) in tensors.items():
        assert actual is not None and expected is not None
        max_absolute, max_relative = error_metrics(actual, expected)
        row[f"{name}_max_abs_error"] = max_absolute
        row[f"{name}_max_rel_error"] = max_relative
        passed = passed and torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    row["passed"] = bool(passed)
    return row


def main() -> None:
    args = parse_args()
    environment = configure_cuda_environment(require_rtx4090=True)
    implementations = (
        ("pytorch_tiled", FlashAttentionPyTorch),
        ("triton", FlashAttentionTriton),
    )
    rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        for head_dim in HEAD_DIMS:
            for dtype in DTYPES:
                for is_causal in CAUSAL_VALUES:
                    for implementation_name, implementation in implementations:
                        row = run_case(
                            implementation_name,
                            implementation,
                            seed=seed,
                            head_dim=head_dim,
                            sequence_length=args.sequence_length,
                            dtype=dtype,
                            is_causal=is_causal,
                        )
                        rows.append(row)
                        print(
                            f"{implementation_name=} {seed=} {head_dim=} dtype={row['dtype']} "
                            f"{is_causal=} passed={row['passed']}"
                        )
                    gc.collect()
                    torch.cuda.empty_cache()

    payload = {
        "metadata": environment_metadata(
            environment,
            command="python student_scripts/a2k/run_correctness.py",
            seed=list(SEEDS),
            warmup="not applicable",
            measurement="3 seeds x 3 head dimensions x 2 dtypes x 2 causal settings",
        ),
        "summary": {
            "total": len(rows),
            "passed": sum(bool(row["passed"]) for row in rows),
            "failed": sum(not bool(row["passed"]) for row in rows),
        },
        "results": rows,
    }
    output_path = args.output_dir / "correctness.json"
    write_json(output_path, payload)
    print(f"saved {len(rows)} rows to {output_path}")
    if payload["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
