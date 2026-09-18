from __future__ import annotations

import argparse
import time

import torch

from cs336_systems.a2k.attention import FlashAttentionTriton
from student_scripts.a2k.common import DTYPES, environment, failure_record, memory_stats, parse_bool, timed_cuda, write_json


def eager(q, k, v, causal):
    scores = q @ k.transpose(-1, -2) * (q.shape[-1] ** -0.5)
    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def main(args):
    stage = "setup"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        dtype = DTYPES[args.dtype]
        q = torch.randn(args.batch_size, args.sequence_length, args.head_dim, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        do = torch.randn_like(q)
        implementations = {"eager": eager, "triton": FlashAttentionTriton.apply}
        compile_unsupported = tuple(__import__("sys").version_info[:2]) >= (3, 13) and tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6)
        if not compile_unsupported:
            implementations["compiled"] = torch.compile(eager, fullgraph=True)
        results = {}
        for implementation, fn in implementations.items():
            cold_ms = None
            if implementation in {"compiled", "triton"}:
                stage = f"{implementation}_cold_start"
                torch.cuda.synchronize()
                start = time.perf_counter()
                fn(q, k, v, args.causal)
                torch.cuda.synchronize()
                cold_ms = (time.perf_counter() - start) * 1000

            def forward(fn=fn):
                return fn(q, k, v, args.causal)

            def backward():
                q.grad = k.grad = v.grad = None
                forward().backward(do)

            metrics = {}
            for name, call in (("forward", forward), ("backward", backward), ("forward_backward", backward)):
                stage = f"{implementation}_{name}"
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                metrics[name] = timed_cuda(call, args.warmup_ms, args.rep_ms)
                metrics[name].update(memory_stats())
            results[implementation] = {"cold_start_ms": cold_ms, "metrics": metrics}
        for mode in ("forward", "backward", "forward_backward"):
            base = results["eager"]["metrics"][mode]["p50_ms"]
            for implementation in results:
                results[implementation]["metrics"][mode]["speedup_vs_eager"] = base / results[implementation]["metrics"][mode]["p50_ms"]
        write_json(args.output, {"status": "ok", "config": vars(args), "environment": environment(), "compile_status": "unsupported" if compile_unsupported else "available", "implementations": results})
    except Exception as exc:  # noqa: BLE001 - experiment failures must be serialized.
        write_json(args.output, failure_record(args, exc, stage))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--sequence-length", type=int, required=True)
    p.add_argument("--head-dim", type=int, choices=[64, 128], required=True)
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--causal", type=parse_bool, required=True)
    p.add_argument("--warmup-ms", type=float, default=100)
    p.add_argument("--rep-ms", type=float, default=300)
    p.add_argument("--output", required=True)
    main(p.parse_args())
