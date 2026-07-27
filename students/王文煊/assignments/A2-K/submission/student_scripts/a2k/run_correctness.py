"""A2-K task 5 (section 9.1): extended correctness matrix.

3 seeds x head_dim {32, 64, 128} x {causal, non-causal}, validating O, L,
dQ, dK, dV of the student Triton FlashAttention-2 (and the pure-PyTorch
tiled reference) against an explicit reference implementation. At least one
FP32 configuration runs with TF32 disabled.

Run:
    PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python student_scripts/a2k/run_correctness.py
"""

from __future__ import annotations

import math
import os
import sys

import torch
from einops import einsum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import collect_metadata, git_commit, set_allocator_limit, write_json  # noqa: E402

ALLOCATOR_FRACTION = set_allocator_limit()  # before any CUDA allocation

from cs336_systems.a2k.flash_pytorch import FlashAttentionPyTorch  # noqa: E402
from cs336_systems.a2k.flash_triton import FlashAttentionTriton  # noqa: E402

OUT_DIR = os.path.join("local_results", "a2k")
BATCH = 2
SEQ = 256
SEEDS = [0, 1, 2]
DS = [32, 64, 128]
RTOL = 1e-2
ATOL = 1e-2


def reference_attention_and_lse(q, k, v, is_causal):
    nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    s = einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(d)
    if is_causal:
        qi = torch.arange(nq, device=q.device)[:, None]
        ki = torch.arange(nk, device=q.device)[None, :]
        s = s.masked_fill(ki > qi, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = einsum(p, v, "... q k, ... k d -> ... q d")
    lse = torch.logsumexp(s, dim=-1)
    return o, lse


def rel_err(a, b):
    return ((a - b).abs() / (b.abs() + 1e-6)).max().item()


def run_config(impl_cls, seed, d, is_causal, dtype, tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.manual_seed(seed)
    q = torch.randn(BATCH, SEQ, d, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(BATCH, SEQ, d, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(BATCH, SEQ, d, device="cuda", dtype=dtype, requires_grad=True)
    do = torch.randn(BATCH, SEQ, d, device="cuda", dtype=dtype)

    q2 = q.detach().clone().double().requires_grad_(True)
    k2 = k.detach().clone().double().requires_grad_(True)
    v2 = v.detach().clone().double().requires_grad_(True)
    o_ref, l_ref = reference_attention_and_lse(q2, k2, v2, is_causal)
    o_ref.backward(do.double())

    o = impl_cls.apply(q, k, v, is_causal)
    lse_saved = [t for t in o.grad_fn.saved_tensors if t.shape == (BATCH, SEQ)][0]
    o.backward(do)

    checks = {
        "O": (o, o_ref),
        "L": (lse_saved, l_ref),
        "dQ": (q.grad, q2.grad),
        "dK": (k.grad, k2.grad),
        "dV": (v.grad, v2.grad),
    }
    entry = {
        "impl": impl_cls.__name__,
        "seed": seed,
        "batch": BATCH,
        "seq_len": SEQ,
        "head_dim": d,
        "dtype": str(dtype).replace("torch.", ""),
        "tf32": tf32,
        "is_causal": is_causal,
        "rtol": RTOL,
        "atol": ATOL,
        "tensors": {},
        "pass": True,
    }
    for name, (got, ref) in checks.items():
        got_d = got.detach().double()
        max_abs = (got_d - ref).abs().max().item()
        max_rel = rel_err(got_d, ref)
        ok = bool(torch.allclose(got_d, ref, rtol=RTOL, atol=ATOL))
        entry["tensors"][name] = {
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "pass": ok,
        }
        entry["pass"] = entry["pass"] and ok
    return entry


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = []
    for impl_cls in [FlashAttentionTriton, FlashAttentionPyTorch]:
        for seed in SEEDS:
            for d in DS:
                for is_causal in [False, True]:
                    # FP32 + TF32 off for seed 0 / d 64; BF16 elsewhere
                    if seed == 0 and d == 64:
                        dtype, tf32 = torch.float32, False
                    else:
                        dtype, tf32 = torch.bfloat16, True
                    e = run_config(impl_cls, seed, d, is_causal, dtype, tf32)
                    entries.append(e)
                    print(impl_cls.__name__, seed, d, is_causal, e["pass"])
    n_pass = sum(1 for e in entries if e["pass"])
    result = {
        "metadata": collect_metadata(
            {
                "script": "student_scripts/a2k/run_correctness.py",
                "command": "PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 .venv/bin/python student_scripts/a2k/run_correctness.py",
                "commit": git_commit(),
            }
        ),
        "reference": "explicit softmax attention in float64 (tests/_attention_and_lse logic)",
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "num_configs": len(entries),
        "num_pass": n_pass,
        "num_fail": len(entries) - n_pass,
        "entries": entries,
    }
    path = os.path.join(OUT_DIR, "correctness.json")
    write_json(result, path)
    print("wrote", path, f"{n_pass}/{len(entries)} pass")


if __name__ == "__main__":
    main()
