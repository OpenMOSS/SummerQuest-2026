from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from student_scripts.a2k.common import append_csv, do_bench, peak_memory, require_cuda_and_limit_allocator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--implementation", choices=("eager", "compiled"), required=True)
    parser.add_argument("--phase", choices=("forward", "forward_backward"), required=True)
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/compile_comparison.csv"))
    args = parser.parse_args()
    require_cuda_and_limit_allocator()
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, args.sequence_length, args.head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    def eager(q, k, v):
        return explicit_attention(q, k, v, True)
    function = eager if args.implementation == "eager" else torch.compile(eager)
    grad = torch.randn_like(q)
    def operation():
        output = function(q, k, v)
        if args.phase == "forward_backward":
            output.backward(grad)
            q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - start) * 1000
    torch.cuda.reset_peak_memory_stats()
    quantiles = do_bench(operation)
    append_csv(args.output, {
        "scope": "attention", "model_size": "", "implementation": args.implementation, "batch_size": 1,
        "sequence_length": args.sequence_length, "head_dim": args.head_dim, "dtype": "bf16", "causal": True,
        "phase": args.phase, "cold_start_ms": cold_start_ms, **quantiles, "samples_ms": "",
        **peak_memory(), "status": "success",
    })


if __name__ == "__main__":
    main()
