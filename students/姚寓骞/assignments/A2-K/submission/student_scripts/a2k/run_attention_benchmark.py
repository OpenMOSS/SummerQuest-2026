from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from cs336_systems.a2k.flash_attention import FlashAttentionTriton
from student_scripts.a2k.common import append_csv, do_bench, peak_memory, require_cuda_and_limit_allocator


def measure(function, q, k, v, phase: str) -> tuple[dict[str, float], dict]:
    grad = torch.randn_like(q)
    if phase == "backward":
        output = function(q, k, v)
        def operation():
            output.backward(grad, retain_graph=True)
            q.grad = k.grad = v.grad = None
    else:
        def operation():
            if phase == "forward":
                function(q, k, v)
                return
            output = function(q, k, v)
            output.backward(grad)
            q.grad = k.grad = v.grad = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    quantiles = do_bench(operation)
    return quantiles, peak_memory()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), required=True)
    parser.add_argument("--implementation", choices=("eager", "compiled", "triton"), required=True)
    parser.add_argument("--phase", choices=("forward", "backward", "forward_backward"), required=True)
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/flash_benchmark.csv"))
    args = parser.parse_args()
    require_cuda_and_limit_allocator()
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, args.sequence_length, args.head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    def eager(q, k, v):
        return explicit_attention(q, k, v, True)
    functions = {"eager": eager, "compiled": torch.compile(eager), "triton": lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True)}
    status, error, quantiles, memory = "success", "", {}, {"peak_allocated_mib": "", "peak_reserved_mib": ""}
    try:
        quantiles, memory = measure(functions[args.implementation], q, k, v, args.phase)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        status, error = ("oom" if isinstance(exc, torch.OutOfMemoryError) else "error"), type(exc).__name__
    row = {
        "implementation": args.implementation, "batch_size": 1, "sequence_length": args.sequence_length,
        "head_dim": args.head_dim, "dtype": "bf16", "causal": True, "phase": args.phase,
        "warmup_ms": 100, "measurement_ms": 300,
        **(quantiles if quantiles else {"p20_ms": "", "p50_ms": "", "p80_ms": ""}),
        **memory, "speedup_vs_eager": "", "status": status,
        "error_type": error, "block_q": 64 if args.implementation == "triton" else "",
        "block_k": 64 if args.implementation == "triton" else "", "num_warps": 4 if args.implementation == "triton" else "",
        "num_stages": 2 if args.implementation == "triton" else "",
        "command": " ".join(sys.argv),
    }
    append_csv(args.output, row)


if __name__ == "__main__":
    main()
