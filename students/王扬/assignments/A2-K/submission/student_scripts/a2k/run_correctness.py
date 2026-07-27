from __future__ import annotations

import argparse

import torch

from cs336_systems.a2k import FlashAttentionPyTorchFunction, FlashAttentionTritonFunction
from student_scripts.a2k.common import add_common_args, ensure_dirs, set_allocator_limit, write_json


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) * (d**-0.5)
    if is_causal:
        q_idx = torch.arange(q.shape[-2], device=q.device)
        k_idx = torch.arange(k.shape[-2], device=q.device)
        scores = scores.masked_fill(~(q_idx[:, None] >= k_idx[None, :]), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v, torch.logsumexp(scores, dim=-1)


def max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    a = actual.detach().float()
    e = expected.detach().float()
    abs_err = (a - e).abs().max().item()
    rel_err = ((a - e).abs() / e.abs().clamp_min(1e-8)).max().item()
    return abs_err, rel_err


def run_case(fn, seed: int, head_dim: int, dtype: torch.dtype, is_causal: bool, device: torch.device):
    torch.manual_seed(seed)
    batch, n_queries, n_keys = 2, 96, 96
    q = torch.randn(batch, n_queries, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, n_keys, head_dim, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(batch, n_queries, head_dim, device=device, dtype=dtype)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out = fn.apply(q, k, v, is_causal)
    lse = [t for t in out.grad_fn.saved_tensors if t.shape == (batch, n_queries)][0]
    out_ref, lse_ref = reference(q_ref, k_ref, v_ref, is_causal)
    out.backward(grad)
    out_ref.backward(grad)

    tolerance = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    out_abs, out_rel = max_errors(out, out_ref)
    lse_abs, lse_rel = max_errors(lse, lse_ref)
    dq_abs, dq_rel = max_errors(q.grad, q_ref.grad)
    dk_abs, dk_rel = max_errors(k.grad, k_ref.grad)
    dv_abs, dv_rel = max_errors(v.grad, v_ref.grad)
    status = "pass" if max(out_abs, lse_abs, dq_abs, dk_abs, dv_abs) <= tolerance * 10 else "fail"
    return {
        "seed": seed,
        "batch": batch,
        "heads": 1,
        "seq_q": n_queries,
        "seq_k": n_keys,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "causal": is_causal,
        "output_max_abs": out_abs,
        "output_max_rel": out_rel,
        "lse_max_abs": lse_abs,
        "lse_max_rel": lse_rel,
        "dq_max_abs": dq_abs,
        "dq_max_rel": dq_rel,
        "dk_max_abs": dk_abs,
        "dk_max_rel": dk_rel,
        "dv_max_abs": dv_abs,
        "dv_max_rel": dv_rel,
        "tolerance": tolerance,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    ensure_dirs()
    allocator = set_allocator_limit()
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = []
    seeds = [args.seed, args.seed + 1, args.seed + 2]
    for fn_name, fn in [("pytorch", FlashAttentionPyTorchFunction), ("triton", FlashAttentionTritonFunction)]:
        if fn_name == "triton" and device.type != "cuda":
            continue
        for seed in seeds:
            for head_dim in [32, 64, 128]:
                for is_causal in [False, True]:
                    dtype = torch.float32 if seed == seeds[0] and head_dim == 32 else torch.bfloat16
                    old_tf32 = torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None
                    if dtype == torch.float32 and torch.cuda.is_available():
                        torch.backends.cuda.matmul.allow_tf32 = False
                    row = run_case(fn, seed, head_dim, dtype, is_causal, device)
                    row["implementation"] = fn_name
                    rows.append(row)
                    if old_tf32 is not None:
                        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    write_json(args.output_dir / "correctness.json", {"allocator": allocator, "results": rows})


if __name__ == "__main__":
    main()
