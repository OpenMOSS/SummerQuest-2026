from __future__ import annotations

import argparse
from pathlib import Path

import torch

from student_scripts.a2k.benchmark_attention import explicit_attention
from tests.adapters import get_flashattention_autograd_function_triton
from .common import allocator_guard, append_csv, memory_stats, quantiles, reset_peak, sync, timed


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=0); p.add_argument("--seqs", default="512,2048,8192,16384"); p.add_argument("--dims", default="64,128"); p.add_argument("--warmup", type=int, default=100); p.add_argument("--rep", type=int, default=300); p.add_argument("--implementations", default="eager,compiled,triton"); p.add_argument("--phases", default="forward,backward,forward-backward"); p.add_argument("--query-tile", type=int, default=32); p.add_argument("--key-tile", type=int, default=64); p.add_argument("--num-warps", type=int, default=4); p.add_argument("--num-stages", type=int, default=1)
    args = p.parse_args(); allocator_guard(args.device)
    impls = [("eager", None), ("compiled", None), ("triton", get_flashattention_autograd_function_triton)]
    impls = [item for item in impls if item[0] in set(args.implementations.split(","))]
    fields = ["implementation", "seq_len", "head_dim", "phase", "dtype", "query_tile", "key_tile", "num_warps", "num_stages", "cold_start_ms", "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib", "peak_reserved_mib", "status", "error", "speedup_vs_eager"]
    eager_p50: dict[tuple[int, int, str], float] = {}
    for n in map(int, args.seqs.split(",")):
        for d in map(int, args.dims.split(",")):
            for name, getter in impls:
                try: fn_cls = getter() if getter else None
                except Exception as e: fn_cls = None
                for phase in args.phases.split(","):
                    row = {"implementation": name, "seq_len": n, "head_dim": d, "phase": phase, "dtype": "bf16", "query_tile": args.query_tile if name == "triton" else "", "key_tile": args.key_tile if name == "triton" else "", "num_warps": args.num_warps if name == "triton" else "", "num_stages": args.num_stages if name == "triton" else "", "cold_start_ms": "", "status": "ok", "error": "", "speedup_vs_eager": ""}
                    try:
                        # Re-seeding before each implementation makes the
                        # inputs identical for a given shape and phase.
                        phase_index = {"forward": 0, "backward": 1, "forward-backward": 2}[phase]
                        torch.manual_seed(args.seed + n * 1000 + d * 10 + phase_index)
                        q = torch.randn(1, n, d, device=args.device, dtype=torch.bfloat16, requires_grad=phase != "forward"); k = torch.randn_like(q, requires_grad=phase != "forward"); v = torch.randn_like(q, requires_grad=phase != "forward")
                        if name == "triton" and fn_cls is None: raise RuntimeError("implementation unavailable")
                        f = (lambda: explicit_attention(q, k, v, True)) if name in ("eager", "compiled") else (lambda: fn_cls.apply(q, k, v, True))
                        if name == "compiled":
                            import time
                            f = torch.compile(f, mode="reduce-overhead")
                            t0 = time.perf_counter(); f();
                            if args.device.startswith("cuda"): torch.cuda.synchronize()
                            row["cold_start_ms"] = (time.perf_counter() - t0) * 1e3
                        if phase == "forward":
                            run = f
                        elif phase == "forward-backward":
                            def run_fwd_bwd():
                                q.grad = k.grad = v.grad = None
                                out = f()
                                out.sum().backward()
                            run = run_fwd_bwd
                        else:
                            run = None
                        if name == "compiled" and phase == "forward-backward":
                            def run_compiled_fwd_bwd():
                                q.grad = k.grad = v.grad = None
                                f().sum().backward()
                            run = run_compiled_fwd_bwd
                        reset_peak(args.device)
                        if phase != "backward":
                            vals = timed(run, args.device, args.warmup, args.rep)
                        else:
                            import time
                            vals = []
                            for i in range(args.warmup + args.rep):
                                q.grad = k.grad = v.grad = None
                                out = f()
                                sync(args.device)
                                t0 = time.perf_counter()
                                out.sum().backward()
                                sync(args.device)
                                if i >= args.warmup:
                                    vals.append((time.perf_counter() - t0) * 1e3)
                        row.update(quantiles(vals)); row.update(memory_stats())
                        key = (n, d, phase)
                        if name == "eager":
                            eager_p50[key] = float(row["p50_ms"]); row["speedup_vs_eager"] = 1.0
                        elif key in eager_p50 and float(row["p50_ms"]) > 0:
                            row["speedup_vs_eager"] = eager_p50[key] / float(row["p50_ms"])
                    except (RuntimeError, torch.OutOfMemoryError) as e:
                        row["status"] = "oom" if "out of memory" in str(e).lower() else "error"; row["error"] = str(e)[:160]
                    append_csv(args.output, row, fields)


if __name__ == "__main__": main()
