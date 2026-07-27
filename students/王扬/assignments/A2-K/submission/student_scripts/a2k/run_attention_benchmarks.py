from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch

from cs336_systems.a2k import FlashAttentionTritonFunction, explicit_attention
from student_scripts.a2k.common import add_common_args, ensure_dirs, peak_memory_mib, quantiles_ms, require_cuda, reset_peak, set_allocator_limit, write_csv


def bench(fn: Callable[[], None], warmup_ms: int = 100, rep_ms: int = 300) -> tuple[float, float, float]:
    try:
        import triton

        p20, p50, p80 = triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, quantiles=[0.2, 0.5, 0.8])
        return float(p20), float(p50), float(p80)
    except Exception:
        samples = []
        end_time = torch.cuda.Event(enable_timing=True)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end_time.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end_time))
        return quantiles_ms(samples)


def bench_backward_only(forward: Callable[[], torch.Tensor], grad: torch.Tensor, params: tuple[torch.Tensor, ...], warmup_steps: int = 20, rep: int = 50):
    def clear_grads() -> None:
        for param in params:
            param.grad = None

    for _ in range(warmup_steps):
        clear_grads()
        out = forward()
        out.backward(grad)
    torch.cuda.synchronize()

    samples = []
    for _ in range(rep):
        clear_grads()
        out = forward()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out.backward(grad)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return quantiles_ms(samples)


def make_inputs(sequence_length: int, head_dim: int, dtype: torch.dtype, device: torch.device):
    q = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, sequence_length, head_dim, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn_like(q)
    return q, k, v, grad


def run_phase(impl: str, sequence_length: int, head_dim: int, phase: str, device: torch.device):
    dtype = torch.bfloat16
    q, k, v, grad = make_inputs(sequence_length, head_dim, dtype, device)
    is_causal = True
    compiled_fn = None
    if impl == "compiled":
        compiled_fn = torch.compile(explicit_attention)

    def forward():
        if impl == "eager":
            return explicit_attention(q, k, v, is_causal)
        if impl == "compiled":
            return compiled_fn(q, k, v, is_causal)
        return FlashAttentionTritonFunction.apply(q, k, v, is_causal)

    def forward_backward():
        q.grad = k.grad = v.grad = None
        out = forward()
        out.backward(grad, retain_graph=False)

    if phase == "forward":
        fn = forward
    elif phase == "backward":
        fn = None
    else:
        fn = forward_backward

    reset_peak()
    try:
        if phase == "backward":
            p20, p50, p80 = bench_backward_only(forward, grad, (q, k, v))
        else:
            p20, p50, p80 = bench(fn)
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        p20 = p50 = p80 = math.nan
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "OOM"
    except Exception:
        p20 = p50 = p80 = math.nan
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "fail"

    block_m = block_n = warps = stages = "NA"
    if impl == "triton":
        block_m, block_n, _block_d, warps, stages = 16, 32, head_dim, 4, 3
    return {
        "implementation": impl,
        "batch_size": 1,
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": "bf16",
        "causal": True,
        "phase": phase,
        "warmup_ms": 100,
        "rep_ms": 300,
        "p20_ms": p20,
        "p50_ms": p50,
        "p80_ms": p80,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "speedup_vs_eager": "NA",
        "status": status,
        "triton_query_tile": block_m,
        "triton_key_tile": block_n,
        "triton_num_warps": warps,
        "triton_num_stages": stages,
    }


def add_speedups(rows: list[dict]) -> None:
    eager = {
        (r["sequence_length"], r["head_dim"], r["phase"]): r
        for r in rows
        if r["implementation"] == "eager" and r["status"] == "ok"
    }
    for row in rows:
        key = (row["sequence_length"], row["head_dim"], row["phase"])
        base = eager.get(key)
        if row["implementation"] == "eager" and row["status"] == "ok":
            row["speedup_vs_eager"] = 1.0
        elif base is not None and row["status"] == "ok" and row["p50_ms"]:
            row["speedup_vs_eager"] = float(base["p50_ms"]) / float(row["p50_ms"])


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--mode", choices=["baseline", "flash"], default="flash")
    args = parser.parse_args()

    ensure_dirs()
    set_allocator_limit()
    device = require_cuda()
    torch.manual_seed(args.seed)

    rows = []
    if args.mode == "baseline":
        implementations = ["eager"]
        shapes = [(s, d) for s in [512, 2048, 8192] for d in [64, 128]]
        output = args.output_dir / "attention_baseline.csv"
        fieldnames = [
            "implementation",
            "batch_size",
            "sequence_length",
            "head_dim",
            "dtype",
            "causal",
            "phase",
            "warmup_ms",
            "rep_ms",
            "timer",
            "p20_ms",
            "p50_ms",
            "p80_ms",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "status",
        ]
    else:
        implementations = ["eager", "compiled", "triton"]
        shapes = [(s, d) for s in [512, 2048, 8192] for d in [64, 128]]
        shapes += [(16384, 64), (16384, 128)]
        output = args.output_dir / "flash_benchmark.csv"
        fieldnames = [
            "implementation",
            "batch_size",
            "sequence_length",
            "head_dim",
            "dtype",
            "causal",
            "phase",
            "warmup_ms",
            "rep_ms",
            "p20_ms",
            "p50_ms",
            "p80_ms",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "speedup_vs_eager",
            "status",
            "triton_query_tile",
            "triton_key_tile",
            "triton_num_warps",
            "triton_num_stages",
        ]

    for sequence_length, head_dim in shapes:
        for impl in implementations:
            if args.mode == "flash" and sequence_length == 16384 and impl == "compiled":
                continue
            for phase in ["forward", "backward", "forward-backward"]:
                row = run_phase(impl, sequence_length, head_dim, phase, device)
                if args.mode == "baseline":
                    row["timer"] = "triton.testing.do_bench"
                    row = {k: row[k] for k in fieldnames}
                rows.append(row)

    if args.mode == "flash":
        add_speedups(rows)
    write_csv(output, rows, fieldnames)


if __name__ == "__main__":
    main()
