from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .common import add_common_args, allocator_guard, append_csv, memory_stats, quantiles, reset_peak, sync, timed


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    d = q.shape[-1]
    s = torch.matmul(q, k.transpose(-1, -2)) * (d**-0.5)
    if causal:
        nq, nk = s.shape[-2:]
        mask = torch.tril(torch.ones((nq, nk), device=s.device, dtype=torch.bool))
        s = s.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(s, dim=-1), v)


def main() -> None:
    p = argparse.ArgumentParser(); add_common_args(p)
    p.add_argument("--seqs", default="512,2048,8192")
    p.add_argument("--dims", default="64,128")
    p.add_argument("--phases", default="forward,backward,forward-backward")
    p.add_argument("--implementation", choices=["eager", "compiled"], default="eager")
    p.add_argument("--warmup", type=int, default=10); p.add_argument("--rep", type=int, default=30)
    args = p.parse_args(); device = args.device
    torch.manual_seed(args.seed); guard = allocator_guard(device)
    fields = ["implementation", "seq_len", "head_dim", "dtype", "phase", "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib", "peak_reserved_mib", "status", "warmup_steps", "measurement_steps", "quantiles", "causal", "error"]
    for n in map(int, args.seqs.split(",")):
        for d in map(int, args.dims.split(",")):
            for phase in args.phases.split(","):
                row = {"implementation": args.implementation, "seq_len": n, "head_dim": d, "dtype": "bf16", "phase": phase, "status": "ok", "warmup_steps": args.warmup, "measurement_steps": args.rep, "quantiles": "0.2,0.5,0.8", "causal": True, "error": ""}
                try:
                    q = torch.randn((1, n, d), device=device, dtype=torch.bfloat16, requires_grad=phase != "forward")
                    k = torch.randn_like(q, requires_grad=phase != "forward"); v = torch.randn_like(q, requires_grad=phase != "forward")
                    fn = lambda: explicit_attention(q, k, v, True)
                    if args.implementation == "compiled" and hasattr(torch, "compile"):
                        fn = torch.compile(fn, mode="max-autotune")
                        fn()  # cold compile outside timed region
                    reset_peak(device)
                    if phase == "forward":
                        vals = timed(fn, device, args.warmup, args.rep)
                    elif phase == "forward-backward":
                        def run_fwd_bwd():
                            q.grad = k.grad = v.grad = None
                            fn().sum().backward()
                        vals = timed(run_fwd_bwd, device, args.warmup, args.rep)
                    else:
                        import time
                        vals = []
                        for i in range(args.warmup + args.rep):
                            q.grad = k.grad = v.grad = None
                            out = fn()
                            sync(device)
                            t0 = time.perf_counter()
                            out.sum().backward()
                            sync(device)
                            if i >= args.warmup:
                                vals.append((time.perf_counter() - t0) * 1e3)
                    row.update(quantiles(vals)); row.update(memory_stats())
                except (RuntimeError, torch.OutOfMemoryError) as e:
                    row.update({"status": "oom" if "out of memory" in str(e).lower() else "error", "error": str(e)[:200]})
                    print(f"{args.implementation} S={n} D={d} phase={phase}: {type(e).__name__}: {e}", flush=True)
                append_csv(args.output, row, fields)


if __name__ == "__main__":
    main()
