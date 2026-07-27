"""Benchmark explicit PyTorch attention for one shape and phase."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from student_scripts.a2k.common import make_metadata, measure, parse_dtype, run_safe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--phase", choices=("forward", "backward", "forward_backward"), required=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = make_metadata(args.seed, " ".join(__import__("sys").argv))
    metadata["experiment"] = {
        "implementation": "eager_explicit",
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim,
        "batch_size": 1,
        "dtype": args.dtype,
        "causal": True,
        "phase": args.phase,
    }

    def body() -> dict[str, object]:
        dtype = parse_dtype(args.dtype)
        q = torch.randn(
            (1, args.sequence_length, args.head_dim),
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        do = torch.randn_like(q)

        if args.phase == "forward":

            def fn() -> None:
                explicit_attention(q, k, v, True)

            reset = None
        elif args.phase == "backward":
            output = explicit_attention(q, k, v, True)

            def fn() -> None:
                torch.autograd.backward(output, do, retain_graph=True)

            reset = (q, k, v)
        else:

            def fn() -> None:
                output = explicit_attention(q, k, v, True)
                output.backward(do)

            reset = (q, k, v)
        stats = measure(fn, warmup_ms=100, rep_ms=300, reset_grads=reset)
        return {
            "implementation": "eager_explicit",
            "sequence_length": args.sequence_length,
            "head_dim": args.head_dim,
            "batch_size": 1,
            "dtype": args.dtype,
            "causal": True,
            "phase": args.phase,
            **stats,
        }

    return run_safe(args.output, metadata, body)


if __name__ == "__main__":
    raise SystemExit(main())
