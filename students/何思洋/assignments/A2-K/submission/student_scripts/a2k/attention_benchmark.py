from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_systems.a2k import FlashAttentionTritonFunction, explicit_attention
from student_scripts.a2k.common import bench, memory_stats, metadata, set_allocator_limit, set_seed, summarize, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max-seq", type=int, default=8192)
    parser.add_argument("--sequence-length", type=int, action="append")
    parser.add_argument("--allocator-limit-mib", type=int, default=23552)
    return parser.parse_args()


def run_phase(impl: str, phase: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, warmup: int, steps: int):
    compiled = None
    if impl == "compiled":
        compiled = torch.compile(lambda a, b, c: explicit_attention(a, b, c, causal)[0], mode="reduce-overhead")

    def forward():
        if impl == "eager":
            return explicit_attention(q, k, v, causal)[0]
        if impl == "compiled":
            return compiled(q, k, v)
        return FlashAttentionTritonFunction.apply(q, k, v, causal)

    def fn():
        for t in (q, k, v):
            t.grad = None
        if phase == "forward":
            with torch.no_grad():
                forward()
            return
        out = forward()
        grad = torch.randn_like(out)
        out.backward(grad)

    return bench(fn, warmup, steps)


def main() -> int:
    args = parse_args()
    allocator = set_allocator_limit(args.allocator_limit_mib)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    sequences = args.sequence_length or [512, 2048, 8192, 16384]
    for seq in sequences:
        if seq > args.max_seq:
            continue
        for d in [64, 128]:
            for impl in ["eager", "compiled", "triton"]:
                if seq == 16384 and impl == "compiled":
                    continue
                for phase in ["forward", "backward", "forward_backward"]:
                    set_seed(2026 + seq + d)
                    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
                    q = torch.randn(1, seq, d, device=device, dtype=dtype, requires_grad=phase != "forward")
                    k = torch.randn(1, seq, d, device=device, dtype=dtype, requires_grad=phase != "forward")
                    v = torch.randn(1, seq, d, device=device, dtype=dtype, requires_grad=phase != "forward")
                    samples, status, error = run_phase(impl, phase, q, k, v, True, args.warmup, args.steps)
                    stats = summarize(samples)
                    rows.append(
                        {
                            "implementation": impl,
                            "sequence_length": seq,
                            "head_dim": d,
                            "batch_size": 1,
                            "dtype": str(dtype),
                            "causal": True,
                            "phase": phase,
                            "warmup": args.warmup,
                            "steps": args.steps,
                            "samples_ms": json.dumps(samples),
                            **stats,
                            **memory_stats(),
                            "triton_block_m": (8 if d >= 128 else 16) if impl == "triton" else "",
                            "triton_block_n": (32 if d >= 128 else 64) if impl == "triton" else "",
                            "num_warps": 4 if impl == "triton" else "",
                            "num_stages": 3 if impl == "triton" else "",
                            "status": status,
                            "error": error,
                        }
                    )
                    del q, k, v
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    write_csv(args.output, rows)
    write_json(args.metadata_output, metadata(vars(args), allocator))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
