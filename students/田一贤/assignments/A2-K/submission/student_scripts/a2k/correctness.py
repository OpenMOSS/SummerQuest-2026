from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import configure_allocator_guard

from cs336_systems.a2k.attention import FlashAttentionPyTorch, _tiled_attention
from .common import dense_attention


def run(seed: int, dim: int, causal: bool, device: torch.device) -> dict:
    torch.manual_seed(seed)
    q = torch.randn(2, 19, dim, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(2, 23, dim, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(2, 23, dim, device=device, dtype=torch.float32, requires_grad=True)
    do = torch.randn_like(q)
    out = FlashAttentionPyTorch.apply(q, k, v, causal)
    tiled_out, tiled_lse = _tiled_attention(q, k, v, causal)
    ref, lse_ref = dense_attention(q, k, v, causal)
    out.backward(do)
    qg, kg, vg = q.grad, k.grad, v.grad
    q0, k0, v0 = (
        q.detach().requires_grad_(),
        k.detach().requires_grad_(),
        v.detach().requires_grad_(),
    )
    ref0, _ = dense_attention(q0, k0, v0, causal)
    ref0.backward(do)
    quantities = {
        "output": (out.detach(), ref.detach()),
        "logsumexp": (tiled_lse.detach(), lse_ref.detach()),
        "dQ": (qg, q0.grad),
        "dK": (kg, k0.grad),
        "dV": (vg, v0.grad),
    }
    # LSE is not returned, so recompute the reference implementation for the
    # explicit scalar error check; the autograd context is checked by tests.
    result = {
        "seed": seed,
        "head_dim": dim,
        "causal": causal,
        "device": str(device),
        "status": "pass",
    }
    for name, (actual, expected) in quantities.items():
        result[f"{name}_max_abs_error"] = float((actual - expected).abs().max())
    result["tolerance"] = {"rtol": 1e-2, "atol": 1e-2}
    passed = bool(
        torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
        and torch.allclose(tiled_out, ref, rtol=1e-2, atol=1e-2)
        and torch.allclose(tiled_lse, lse_ref, rtol=1e-2, atol=1e-2)
        and torch.allclose(qg, q0.grad, rtol=1e-2, atol=1e-2)
        and torch.allclose(kg, k0.grad, rtol=1e-2, atol=1e-2)
        and torch.allclose(vg, v0.grad, rtol=1e-2, atol=1e-2)
    )
    result["pass"] = passed
    result["status"] = "pass" if passed else "fail"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/correctness.json"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    guard = configure_allocator_guard() if args.device.startswith("cuda") else {"applied": False}
    device = torch.device(args.device)
    rows = [
        run(seed, dim, causal, device)
        for seed in (7, 19, 41)
        for dim in (32, 64, 128)
        for causal in (False, True)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measurement_collected": True,
        "evaluation_type": "synthetic_proxy",
        "reference": "dense_attention mathematical reference; not dataset ground truth",
        "allocator_guard": guard,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    if not all(row["pass"] for row in rows):
        raise SystemExit("correctness check failed")


if __name__ == "__main__":
    main()
