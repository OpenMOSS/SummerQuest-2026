from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tests.adapters import get_flashattention_autograd_function_pytorch, get_flashattention_autograd_function_triton

from .common import allocator_guard, write_json


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    s = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)
    if causal:
        nq, nk = s.shape[-2:]
        s = s.masked_fill(~torch.tril(torch.ones((nq, nk), device=q.device, dtype=torch.bool)), float("-inf"))
    l = torch.logsumexp(s, dim=-1)
    return torch.softmax(s, dim=-1) @ v, l


def one(impl, device: str, dtype: torch.dtype, d: int, causal: bool, seed: int):
    torch.manual_seed(seed)
    q = torch.randn(2, 32, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(2, 32, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(2, 32, d, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn_like(q)
    o, l = reference(q, k, v, causal)
    q0, k0, v0 = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()
    out = impl.apply(q0, k0, v0, causal)
    saved_l = [t for t in out.grad_fn.saved_tensors if t.shape == l.shape][0]
    out.backward(do)
    # Reference gradients via autograd.
    o.backward(do)
    vals = [o.detach(), out.detach(), l.detach()]
    atol = 2e-2 if dtype != torch.float32 else 1e-3
    rtol = 2e-2 if dtype != torch.float32 else 1e-3
    def relerr(a, b):
        return ((a - b).abs() / b.abs().clamp_min(1e-12)).max().item()
    errs = {"o_abs": (vals[0] - vals[1]).abs().max().item(), "o_rel": relerr(vals[0], vals[1]),
            "lse_abs": (vals[2] - saved_l).abs().max().item(), "lse_rel": relerr(vals[2], saved_l),
            "dq_abs": (q.grad - q0.grad).abs().max().item(), "dq_rel": relerr(q.grad, q0.grad),
            "dk_abs": (k.grad - k0.grad).abs().max().item(), "dk_rel": relerr(k.grad, k0.grad),
            "dv_abs": (v.grad - v0.grad).abs().max().item(), "dv_rel": relerr(v.grad, v0.grad),
            "atol": atol, "rtol": rtol}
    errs["pass"] = max(errs[k] for k in ("o_abs", "lse_abs", "dq_abs", "dk_abs", "dv_abs")) < atol
    return errs


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--seeds", default="0,1,2")
    args = p.parse_args(); guard = allocator_guard(args.device)
    results = {"assignment": "A2-K", "guard": guard, "cases": []}
    for name, getter in [("pytorch", get_flashattention_autograd_function_pytorch), ("triton", get_flashattention_autograd_function_triton)]:
        try: impl = getter()
        except Exception as e:
            results["cases"].append({"implementation": name, "status": "unavailable", "error": str(e)[:200]}); continue
        for seed in map(int, args.seeds.split(",")):
            for d in (32, 64, 128):
                for causal in (False, True):
                    try: results["cases"].append({"implementation": name, "seed": seed, "head_dim": d, "causal": causal, "dtype": "fp32", "status": "ok", **one(impl, args.device, torch.float32, d, causal, seed)})
                    except Exception as e: results["cases"].append({"implementation": name, "seed": seed, "head_dim": d, "causal": causal, "status": "error", "error": str(e)[:200]})
    write_json(args.output, results)


if __name__ == "__main__": main()
