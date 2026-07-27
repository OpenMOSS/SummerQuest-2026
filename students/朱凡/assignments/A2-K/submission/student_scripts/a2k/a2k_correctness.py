#!/usr/bin/env python3
"""Run the A2-K forward/LSE/backward correctness matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from cs336_systems.a2k.attention import FlashAttentionPyTorch, FlashAttentionTriton
from scripts.measurement import configure_allocator


def reference(q, k, v, causal):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causal:
        positions = torch.arange(q.shape[-2], device=q.device)
        scores = scores.masked_fill(positions[:, None] < positions[None, :], -1e6)
    lse = torch.logsumexp(scores.float(), dim=-1)
    return torch.softmax(scores, dim=-1) @ v, lse


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.detach().float() - expected.detach().float()).abs()
    denominator = expected.detach().float().abs().clamp_min(1e-12)
    return {
        "max_abs_error": float(difference.max()),
        "max_rel_error": float((difference / denominator).max()),
    }


def run_case(implementation, device, seed, dimension, dtype, causal):
    torch.manual_seed(seed)
    q = torch.randn(2, 32, dimension, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 32, dimension, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 32, dimension, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn_like(q)
    output = implementation.apply(q, k, v, causal)
    saved_lse = [tensor for tensor in output.grad_fn.saved_tensors if tensor.shape == (2, 32)]
    if len(saved_lse) != 1:
        raise AssertionError(f"expected exactly one LSE tensor, found {len(saved_lse)}")
    expected_output, expected_lse = reference(q, k, v, causal)
    output.backward(do)
    actual_gradients = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())

    rq = q.detach().clone().requires_grad_(True)
    rk = k.detach().clone().requires_grad_(True)
    rv = v.detach().clone().requires_grad_(True)
    reference_output, _ = reference(rq, rk, rv, causal)
    reference_output.backward(do)
    expected_gradients = (rq.grad, rk.grad, rv.grad)
    # The Triton kernel accumulates in FP32 but writes through the device
    # matrix-multiply path, so its float32 result is expected to differ from
    # the unfused PyTorch reference by a few 1e-3.  Keep a stricter tolerance
    # for the exact PyTorch tiled reference and report the Triton tolerance
    # explicitly instead of hiding the numerical difference.
    tolerances = (1e-2, 1e-2) if implementation is FlashAttentionTriton or dtype != torch.float32 else (1e-4, 1e-4)
    checks = {
        "output": error(output, expected_output),
        "logsumexp": error(saved_lse[0], expected_lse),
        "dQ": error(actual_gradients[0], expected_gradients[0]),
        "dK": error(actual_gradients[1], expected_gradients[1]),
        "dV": error(actual_gradients[2], expected_gradients[2]),
    }
    passed = all(
        values["max_abs_error"] <= tolerances[0] or values["max_rel_error"] <= tolerances[1]
        for values in checks.values()
    )
    return {
        "implementation": implementation.__name__,
        "seed": seed,
        "batch_size": 2,
        "sequence_length": 32,
        "head_dim": dimension,
        "dtype": str(dtype).removeprefix("torch."),
        "causal": causal,
        "atol": tolerances[0],
        "rtol": tolerances[1],
        "checks": checks,
        "status": "pass" if passed else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/correctness.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--dimensions", default="32,64,128")
    parser.add_argument("--implementations", default="pytorch,triton")
    parser.add_argument("--allocator-limit-mib", type=int, default=23552)
    args = parser.parse_args()
    device = torch.device(args.device)
    configure_allocator(device, args.allocator_limit_mib)
    if "triton" in args.implementations.split(",") and device.type != "cuda":
        implementations = [("pytorch", FlashAttentionPyTorch)]
    else:
        implementations = [
            (name, FlashAttentionPyTorch if name == "pytorch" else FlashAttentionTriton)
            for name in args.implementations.split(",")
        ]
    rows = []
    for name, implementation in implementations:
        for seed in map(int, args.seeds.split(",")):
            for dimension in map(int, args.dimensions.split(",")):
                for causal in (False, True):
                    try:
                        row = run_case(implementation, device, seed, dimension, torch.float32, causal)
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                        row = {
                            "implementation": name,
                            "seed": seed,
                            "head_dim": dimension,
                            "dtype": "float32",
                            "causal": causal,
                            "status": "error",
                            "error": str(exc),
                        }
                    row["implementation"] = name
                    rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "passed": sum(row["status"] == "pass" for row in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
