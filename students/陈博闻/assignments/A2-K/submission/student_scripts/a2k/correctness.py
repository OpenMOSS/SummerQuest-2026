from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_systems.a2k import FlashAttentionTorch, FlashAttentionTriton, explicit_attention
from student_scripts.a2k.common import require_cuda, write_json


def max_rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denom = torch.maximum(expected.abs(), torch.full_like(expected, 1e-8))
    return float(((actual - expected).abs() / denom).max().item())


def within_tolerance(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> bool:
    return bool(torch.all((actual - expected).abs() <= atol + rtol * expected.abs()).item())


def check_one(impl, *, seed: int, d: int, is_causal: bool, dtype: torch.dtype, device: torch.device) -> dict:
    torch.manual_seed(seed)
    batch, seq = 2, 128
    q = torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, seq, d, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(batch, seq, d, device=device, dtype=dtype)
    ref_q, ref_k, ref_v = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()

    out = impl.apply(q, k, v, is_causal)
    lse = [t for t in out.grad_fn.saved_tensors if t.shape == (batch, seq)][0]
    out.backward(do)
    ref_out, ref_lse = explicit_attention(ref_q.float(), ref_k.float(), ref_v.float(), is_causal)
    ref_out.backward(do.float())
    atol, rtol = (2e-2, 2e-2) if dtype in (torch.float16, torch.bfloat16) else (1e-4, 1e-4)
    rows = {
        "seed": seed,
        "head_dim": d,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": is_causal,
        "max_abs_o": float((out.float() - ref_out).abs().max().item()),
        "max_rel_o": max_rel_error(out.float(), ref_out),
        "max_abs_lse": float((lse.float() - ref_lse).abs().max().item()),
        "max_rel_lse": max_rel_error(lse.float(), ref_lse),
        "max_abs_dq": float((q.grad.float() - ref_q.grad).abs().max().item()),
        "max_rel_dq": max_rel_error(q.grad.float(), ref_q.grad),
        "max_abs_dk": float((k.grad.float() - ref_k.grad).abs().max().item()),
        "max_rel_dk": max_rel_error(k.grad.float(), ref_k.grad),
        "max_abs_dv": float((v.grad.float() - ref_v.grad).abs().max().item()),
        "max_rel_dv": max_rel_error(v.grad.float(), ref_v.grad),
        "atol": atol,
        "rtol": rtol,
    }
    rows["passed"] = all(
        [
            within_tolerance(out.float(), ref_out, atol, rtol),
            within_tolerance(lse.float(), ref_lse, atol, rtol),
            within_tolerance(q.grad.float(), ref_q.grad, atol, rtol),
            within_tolerance(k.grad.float(), ref_k.grad, atol, rtol),
            within_tolerance(v.grad.float(), ref_v.grad, atol, rtol),
        ]
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/correctness.json"))
    parser.add_argument("--include-triton", action="store_true")
    args = parser.parse_args()
    device = require_cuda() if args.include_triton else torch.device("cpu")
    impls = {"pytorch": FlashAttentionTorch}
    if args.include_triton:
        impls["triton"] = FlashAttentionTriton
    results = []
    for name, impl in impls.items():
        for seed in [0, 1, 2]:
            for d in [32, 64, 128]:
                for causal in [False, True]:
                    for dtype in ([torch.float32, torch.bfloat16] if args.include_triton else [torch.float32]):
                        row = check_one(impl, seed=seed, d=d, is_causal=causal, dtype=dtype, device=device)
                        row["implementation"] = name
                        results.append(row)
    write_json(args.output, {"results": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
