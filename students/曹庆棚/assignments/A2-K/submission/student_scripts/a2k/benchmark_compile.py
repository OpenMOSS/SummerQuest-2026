from __future__ import annotations

import argparse
from pathlib import Path

import torch
from cs336_basics.model import BasicsTransformerLM

from .benchmark_attention import explicit_attention
from .common import allocator_guard, append_csv, memory_stats, quantiles, timed


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--warmup", type=int, default=100); p.add_argument("--rep", type=int, default=300)
    args = p.parse_args(); allocator_guard(args.device)
    rows = ["attention-512x64", "attention-2048x128", "attention-8192x128"]
    fields = ["model", "implementation", "seq_len", "head_dim", "phase", "cold_start_ms", "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib", "peak_reserved_mib", "status", "error"]
    for label in rows:
        n, d = map(int, label.split("-")[1].split("x"))
        for impl in ("eager", "compiled"):
            for phase in ("forward", "forward-backward", "train-step"):
                row = {"model": "attention", "implementation": impl, "seq_len": n, "head_dim": d, "phase": phase, "status": "ok", "error": ""}
                try:
                    q = torch.randn(1, n, d, device=args.device, dtype=torch.bfloat16, requires_grad=phase != "forward"); k = torch.randn_like(q, requires_grad=phase != "forward"); v = torch.randn_like(q, requires_grad=phase != "forward")
                    fn = lambda: explicit_attention(q, k, v, True)
                    cold = float("nan")
                    if impl == "compiled" and hasattr(torch, "compile"):
                        cfn = torch.compile(fn, mode="reduce-overhead")
                        import time
                        t0 = time.perf_counter()
                        cold_out = cfn()
                        if phase != "forward":
                            cold_out.sum().backward()
                            q.grad = k.grad = v.grad = None
                        torch.cuda.synchronize() if args.device.startswith("cuda") else None
                        cold = (time.perf_counter()-t0)*1e3; fn = cfn
                    def run_backward():
                        out = fn()
                        (out[0] if isinstance(out, tuple) else out).sum().backward()
                        q.grad = k.grad = v.grad = None
                    run = fn if phase == "forward" else run_backward
                    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
                    row.update({"cold_start_ms": cold, **quantiles(timed(run, args.device, args.warmup, args.rep)), **memory_stats()})
                except (RuntimeError, torch.OutOfMemoryError) as e:
                    row.update({"status": "oom" if "out of memory" in str(e).lower() else "error", "error": str(e)[:240], "cold_start_ms": ""})
                append_csv(args.output, row, fields)

    # Required Stanford-small full-model comparison at S=512.  Keep this in a
    # separate process invocation in the formal protocol; failures are retained.
    model_fields = ["model", "implementation", "seq_len", "head_dim", "phase", "cold_start_ms", "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib", "peak_reserved_mib", "status", "error"]
    cfg = dict(vocab_size=10_000, context_length=512, d_model=768,
               num_layers=12, num_heads=12, d_ff=3072)
    for impl in ("eager", "compiled"):
        for phase in ("forward", "forward-backward", "train-step"):
            row = {"model": "stanford-small", "implementation": impl,
                   "seq_len": 512, "head_dim": 64, "phase": phase,
                   "status": "ok", "cold_start_ms": "", "error": ""}
            try:
                model = BasicsTransformerLM(**cfg).to(args.device)
                x = torch.randint(0, cfg["vocab_size"], (1, 512), device=args.device)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                def eager_step():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                        logits = model(x)
                        loss = logits.float().mean()
                    if phase != "forward": loss.backward()
                    if phase == "train-step": opt.step()
                    return logits
                fn = eager_step
                if impl == "compiled" and hasattr(torch, "compile"):
                    cfn = torch.compile(eager_step, mode="reduce-overhead")
                    import time
                    t0 = time.perf_counter(); cfn();
                    if args.device.startswith("cuda"): torch.cuda.synchronize()
                    row["cold_start_ms"] = (time.perf_counter()-t0)*1e3
                    fn = cfn
                if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
                vals = timed(fn, args.device, args.warmup, args.rep)
                row.update(quantiles(vals), **memory_stats())
            except (RuntimeError, torch.OutOfMemoryError) as e:
                row.update({"status": "oom" if "out of memory" in str(e).lower() else "error", "error": str(e)[:240]})
            append_csv(args.output, row, model_fields)


if __name__ == "__main__": main()
