"""Extended FlashAttention correctness matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_systems.a2k.attention import (
    FlashAttentionPytorch,
    FlashAttentionTriton,
)
from cs336_systems.a2k.runtime import set_allocator_limit, write_json


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1e-8)
    return {
        "max_abs": float(diff.max().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def within_tolerance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> bool:
    """Use the assignment's allclose-style tolerance, not relative error near zero."""

    atol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-2
    return bool(torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocator = set_allocator_limit()
    results: list[dict[str, object]] = []
    for seed in (0, 1, 2):
        for dim in (32, 64, 128):
            for causal in (False, True):
                for dtype in (torch.float32, torch.bfloat16):
                    torch.manual_seed(seed)
                    q = torch.randn((2, 64, dim), device="cuda", dtype=dtype, requires_grad=True)
                    k = torch.randn((2, 64, dim), device="cuda", dtype=dtype, requires_grad=True)
                    v = torch.randn((2, 64, dim), device="cuda", dtype=dtype, requires_grad=True)
                    do = torch.randn_like(q)
                    ref_o, ref_l = _reference(q, k, v, causal)
                    ref_q, ref_k, ref_v = _reference_grads(q, k, v, do, causal)
                    for name, function in (
                        ("pytorch_tiled", FlashAttentionPytorch),
                        ("triton_forward", FlashAttentionTriton),
                    ):
                        case: dict[str, object] = {
                            "seed": seed,
                            "sequence_length": 64,
                            "head_dim": dim,
                            "dtype": str(dtype).replace("torch.", ""),
                            "causal": causal,
                            "implementation": name,
                        }
                        try:
                            q1, k1, v1 = (
                                x.detach().clone().requires_grad_() for x in (q, k, v)
                            )
                            out = function.apply(q1, k1, v1, causal)
                            saved = tuple(out.grad_fn.saved_tensors)
                            saved_lse = next(
                                tensor
                                for tensor in saved
                                if tuple(tensor.shape) == (q.shape[0], q.shape[1])
                            )
                            out.backward(do)
                            if q1.grad is None or k1.grad is None or v1.grad is None:
                                raise RuntimeError("attention backward produced a missing gradient")
                            e_o = errors(out, ref_o)
                            e_l = errors(saved_lse, ref_l)
                            e_q = errors(q1.grad, ref_q)
                            e_k = errors(k1.grad, ref_k)
                            e_v = errors(v1.grad, ref_v)
                            case.update(
                                {
                                    "output": e_o,
                                    "lse": e_l,
                                    "dQ": e_q,
                                    "dK": e_k,
                                    "dV": e_v,
                                    "pass": all(
                                        (
                                            within_tolerance(out, ref_o, dtype),
                                            within_tolerance(saved_lse, ref_l, dtype),
                                            within_tolerance(q1.grad, ref_q, dtype),
                                            within_tolerance(k1.grad, ref_k, dtype),
                                            within_tolerance(v1.grad, ref_v, dtype),
                                        )
                                    ),
                                }
                            )
                        except BaseException as exc:
                            case.update(
                                {
                                    "pass": False,
                                    "status": "oom"
                                    if "out of memory" in str(exc).lower()
                                    else "failed",
                                    "error_type": type(exc).__name__,
                                }
                            )
                        results.append(case)
    write_json(args.output, {
        "allocator": allocator,
        "tolerance": {
            "fp32": {"rtol": 1e-2, "atol": 1e-2},
            "bf16": {"rtol": 2e-2, "atol": 2e-2},
        },
        "cases": results,
        "passed": sum(bool(item["pass"]) for item in results),
        "failed": sum(not bool(item["pass"]) for item in results),
    })
    return 0


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (q.shape[-1] ** 0.5)
    if causal:
        indices = torch.arange(q.shape[1], device=q.device)
        scores = scores.masked_fill(indices[:, None] < indices[None, :], -1e6)
    lse = torch.logsumexp(scores, dim=-1)
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype), lse


def _reference_grads(q, k, v, do, causal):
    q1, k1, v1 = (x.detach().clone().float().requires_grad_() for x in (q, k, v))
    o, _ = _reference(q1, k1, v1, causal)
    o.backward(do.float())
    return q1.grad.to(q.dtype), k1.grad.to(k.dtype), v1.grad.to(v.dtype)


if __name__ == "__main__":
    raise SystemExit(main())
