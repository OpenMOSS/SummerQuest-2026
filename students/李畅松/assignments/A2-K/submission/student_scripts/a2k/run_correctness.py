from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from cs336_systems.a2k.attention import FlashAttentionTriton
from student_scripts.a2k.common import environment, failure_record, write_json


def errors(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "max_rel": (diff / expected.float().abs().clamp_min(1e-6)).max().item(),
    }


def main(args):
    stage = "setup"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cases = []
        for seed in args.seeds:
            for dtype in (torch.float32, torch.bfloat16):
                for causal in (False, True):
                    for d in (32, 64, 128):
                        for n in args.sequence_lengths:
                            stage = f"seed{seed}_n{n}_d{d}_{dtype}_{causal}"
                            torch.manual_seed(seed)
                            q = torch.randn(args.batch_size, n, d, device="cuda", dtype=dtype, requires_grad=True)
                            k = torch.randn_like(q, requires_grad=True)
                            v = torch.randn_like(q, requires_grad=True)
                            do = torch.randn_like(q)
                            expected = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], is_causal=causal)[:, 0]
                            expected.backward(do)
                            expected_grads = (q.grad.detach().clone(), k.grad.detach().clone(), v.grad.detach().clone())
                            q.grad = k.grad = v.grad = None
                            actual = FlashAttentionTriton.apply(q, k, v, causal)
                            actual.backward(do)
                            cases.append({
                                "seed": seed,
                                "sequence_length": n,
                                "head_dim": d,
                                "dtype": str(dtype).removeprefix("torch."),
                                "causal": causal,
                                "output": errors(actual, expected),
                                "dq": errors(q.grad, expected_grads[0]),
                                "dk": errors(k.grad, expected_grads[1]),
                                "dv": errors(v.grad, expected_grads[2]),
                            })
        write_json(args.output, {"status": "ok", "environment": environment(), "cases": cases})
    except Exception as exc:  # noqa: BLE001 - preserve the failed case in JSON.
        write_json(args.output, failure_record(args, exc, stage))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-lengths", type=int, nargs="+", default=[128, 512, 2048])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output", default="local_results/a2k_correctness.json")
    main(p.parse_args())
