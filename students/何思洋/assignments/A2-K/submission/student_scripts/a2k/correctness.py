from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_systems.a2k import FlashAttentionPytorchFunction, FlashAttentionTritonFunction, explicit_attention
from student_scripts.a2k.common import set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def check_case(seed: int, d: int, is_causal: bool, dtype: torch.dtype, device: torch.device) -> dict[str, object]:
    set_seed(seed)
    batch, n = 2, 128
    q = torch.randn(batch, n, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, n, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, n, d, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(batch, n, d, device=device, dtype=dtype)

    ref_out, ref_lse = explicit_attention(q, k, v, is_causal)
    ref_out.backward(grad, retain_graph=True)
    ref_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
    for tensor in (q, k, v):
        tensor.grad = None

    impl = FlashAttentionTritonFunction if device.type == "cuda" else FlashAttentionPytorchFunction
    out = impl.apply(q, k, v, is_causal)
    lse = [t for t in out.grad_fn.saved_tensors if t.shape == (batch, n)][0]
    out.backward(grad)
    got_grads = (q.grad, k.grad, v.grad)

    def err(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
        diff = (a.float() - b.float()).abs()
        denom = b.float().abs().clamp_min(1e-6)
        return float(diff.max().item()), float((diff / denom).max().item())

    checks = {
        "output": err(out, ref_out),
        "lse": err(lse, ref_lse),
        "dq": err(got_grads[0], ref_grads[0]),
        "dk": err(got_grads[1], ref_grads[1]),
        "dv": err(got_grads[2], ref_grads[2]),
    }
    atol = 3e-2 if dtype != torch.float32 else 5e-3
    passed = all(math.isfinite(abs_err) and abs_err <= atol for abs_err, _ in checks.values())
    return {
        "seed": seed,
        "head_dim": d,
        "is_causal": is_causal,
        "dtype": str(dtype),
        "shape": [batch, n, d],
        "tolerance_abs": atol,
        "checks": {name: {"max_abs": value[0], "max_rel": value[1]} for name, value in checks.items()},
        "pass": passed,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cases = []
    for seed in [0, 1, 2]:
        for d in [32, 64, 128]:
            for causal in [False, True]:
                cases.append(check_case(seed, d, causal, torch.float32 if seed == 0 else torch.bfloat16, device))
    write_json(args.output, {"device": str(device), "cases": cases})
    return 0 if all(case["pass"] for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
