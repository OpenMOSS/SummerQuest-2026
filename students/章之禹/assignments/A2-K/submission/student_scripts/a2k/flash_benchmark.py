"""Benchmark one explicit/compiled/Triton FlashAttention configuration."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cs336_systems.a2k.attention import (
    FlashAttentionTriton,
    explicit_attention,
    triton_launch_config,
)
from student_scripts.a2k.common import (
    make_metadata,
    measure,
    parse_dtype,
    run_safe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("eager", "compiled", "triton"), required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--phase", choices=("forward", "backward", "forward_backward"), required=True)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = make_metadata(args.seed, " ".join(__import__("sys").argv))
    launch = triton_launch_config(
        args.sequence_length,
        args.sequence_length,
        args.head_dim,
    )
    metadata["experiment"] = {
        "implementation": args.implementation,
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim,
        "batch_size": 1,
        "dtype": args.dtype,
        "causal": True,
        "phase": args.phase,
        "q_tile_size": (
            launch["q_tile_size"] if args.implementation == "triton" else None
        ),
        "k_tile_size": (
            launch["k_tile_size"] if args.implementation == "triton" else None
        ),
        "num_warps": (
            launch["num_warps"] if args.implementation == "triton" else None
        ),
        "num_stages": (
            launch["num_stages"] if args.implementation == "triton" else None
        ),
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

        if args.implementation == "eager":

            def forward() -> torch.Tensor:
                return explicit_attention(q, k, v, True)
        elif args.implementation == "triton":

            def forward() -> torch.Tensor:
                return FlashAttentionTriton.apply(q, k, v, True)
        else:

            def compile_target() -> torch.Tensor:
                return explicit_attention(q, k, v, True)

            forward = torch.compile(
                compile_target,
                backend="inductor",
                dynamic=False,
            )

        cold_ms = None
        if args.implementation == "compiled":
            torch.cuda.synchronize()
            start = time.perf_counter()
            cold_output = forward()
            if args.phase != "forward":
                cold_output.backward(do, retain_graph=args.phase == "backward")
            torch.cuda.synchronize()
            cold_ms = (time.perf_counter() - start) * 1000.0
            for tensor in (q, k, v):
                tensor.grad = None

        if args.phase == "forward":
            fn = forward
            reset = None
        elif args.phase == "backward":
            output = forward()
            torch.cuda.synchronize()

            def fn() -> None:
                output.backward(do, retain_graph=True)

            reset = (q, k, v)
        else:

            def fn() -> None:
                output = forward()
                output.backward(do)

            reset = (q, k, v)

        stats = measure(fn, warmup_ms=100, rep_ms=300, reset_grads=reset)
        return {
            "implementation": args.implementation,
            "sequence_length": args.sequence_length,
            "head_dim": args.head_dim,
            "batch_size": 1,
            "dtype": args.dtype,
            "causal": True,
            "phase": args.phase,
            "cold_start_ms": cold_ms,
            "q_tile_size": (
                launch["q_tile_size"] if args.implementation == "triton" else None
            ),
            "k_tile_size": (
                launch["k_tile_size"] if args.implementation == "triton" else None
            ),
            "num_warps": (
                launch["num_warps"] if args.implementation == "triton" else None
            ),
            "num_stages": (
                launch["num_stages"] if args.implementation == "triton" else None
            ),
            **stats,
        }

    return run_safe(args.output, metadata, body)


if __name__ == "__main__":
    raise SystemExit(main())
