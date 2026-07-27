from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from cs336_systems.a2k.flash_attention import FlashAttentionPyTorch, FlashAttentionTriton
from student_scripts.a2k.common import metadata, require_cuda_and_limit_allocator, write_json


def errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual - expected).abs()
    relative = delta / expected.abs().clamp_min(1e-6)
    return delta.max().item(), relative.max().item()


def run_case(seed: int, dim: int, causal: bool, dtype: torch.dtype, implementation: str) -> dict:
    torch.manual_seed(seed)
    batch, sequence = 1, 257
    tensors = [torch.randn(batch, sequence, dim, device="cuda", dtype=dtype, requires_grad=True) for _ in range(3)]
    q, k, v = tensors
    grad = torch.randn_like(q)
    reference = explicit_attention(q, k, v, causal)
    ref_grads = torch.autograd.grad(reference, tensors, grad, retain_graph=True)
    cls = FlashAttentionPyTorch if implementation == "pytorch_tiled" else FlashAttentionTriton
    output = cls.apply(q, k, v, causal)
    lse = next(t for t in output.grad_fn.saved_tensors if t.shape == (batch, sequence))
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / dim**0.5
    if causal:
        mask = torch.arange(sequence, device="cuda")[:, None] >= torch.arange(sequence, device="cuda")[None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
    expected_lse = torch.logsumexp(scores, dim=-1)
    actual_grads = torch.autograd.grad(output, tensors, grad)
    fields = {"output": errors(output.float(), reference.float()), "lse": errors(lse, expected_lse)}
    fields |= {name: errors(actual.float(), expected.float()) for name, actual, expected in zip(("dq", "dk", "dv"), actual_grads, ref_grads, strict=True)}
    # BF16 has a 7-bit mantissa. Values around magnitude 4 can differ by one
    # representable step (0.03125), so use a one-step absolute tolerance while
    # retaining a tighter relative tolerance away from zero.
    atol, rtol = (3e-4, 3e-4) if dtype == torch.float32 else (4e-2, 2e-2)
    return {
        "seed": seed, "shape": [batch, sequence, dim], "dtype": str(dtype), "causal": causal,
        "implementation": implementation, "atol": atol, "rtol": rtol,
        "errors": {name: {"max_abs": value[0], "max_rel": value[1]} for name, value in fields.items()},
        "passed": all(abs_error <= atol or rel_error <= rtol for abs_error, rel_error in fields.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/correctness.json"))
    args = parser.parse_args()
    device, fraction = require_cuda_and_limit_allocator()
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    for seed in (0, 1, 2):
        for dim in (32, 64, 128):
            for causal in (False, True):
                dtype = torch.float32 if seed == 0 and dim == 32 else torch.bfloat16
                for implementation in ("pytorch_tiled", "triton"):
                    cases.append(run_case(seed, dim, causal, dtype, implementation))
    write_json(args.output, {"metadata": metadata(device, fraction, 0), "cases": cases})
    if not all(case["passed"] for case in cases):
        raise SystemExit("one or more correctness cases failed")


if __name__ == "__main__":
    main()
