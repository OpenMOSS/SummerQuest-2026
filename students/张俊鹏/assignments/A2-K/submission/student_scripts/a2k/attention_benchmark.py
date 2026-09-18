from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
import triton.testing

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "cs336-basics"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cs336_systems.a2k.explicit_attention import explicit_attention
from cs336_systems.a2k.flash_attention_triton import FlashAttentionTriton


ALLOCATOR_LIMIT_MIB = 23 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-K explicit attention benchmark")
    parser.add_argument("--implementation", choices=("eager", "compiled", "triton"), required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_allocator_limit() -> float:
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, ALLOCATOR_LIMIT_MIB * 1024**2 / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def reset_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def measure_phase(callable_, warmup_ms: int, rep_ms: int) -> tuple[float, float, float, float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    p20, p50, p80 = triton.testing.do_bench(
        callable_,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )
    torch.cuda.synchronize()
    return (
        p20,
        p50,
        p80,
        torch.cuda.max_memory_allocated() / 1024**2,
        torch.cuda.max_memory_reserved() / 1024**2,
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.sequence_length, args.head_dim, args.batch_size, args.warmup_ms, args.rep_ms) <= 0:
        raise ValueError("all size and timing arguments must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    allocator_fraction = configure_allocator_limit()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.bfloat16
    q = torch.randn(args.batch_size, args.sequence_length, args.head_dim, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    do = torch.randn_like(q)

    attention = explicit_attention
    cold_start_ms: float | None = None
    if args.implementation == "compiled":
        attention = torch.compile(explicit_attention, fullgraph=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        attention(q, k, v, args.causal)
        torch.cuda.synchronize()
        cold_start_ms = (time.perf_counter() - start) * 1_000
    elif args.implementation == "triton":
        attention = FlashAttentionTriton.apply

    rows = []
    for phase in ("forward", "backward", "forward_backward"):
        if phase == "forward":
            callable_ = lambda: attention(q, k, v, args.causal)
        elif phase == "backward":
            # Build this graph outside the latency interval. retain_graph=True
            # makes the same graph reusable across do_bench repetitions.
            retained_output = attention(q, k, v, args.causal)

            def callable_() -> None:
                reset_gradients(q, k, v)
                retained_output.backward(do, retain_graph=True)

        else:

            def callable_() -> None:
                reset_gradients(q, k, v)
                output = attention(q, k, v, args.causal)
                output.backward(do)

        row = {
            "implementation": args.implementation,
            "sequence_length": args.sequence_length,
            "head_dim": args.head_dim,
            "batch_size": args.batch_size,
            "dtype": "bfloat16",
            "causal": args.causal,
            "phase": phase,
            "cold_start_ms": cold_start_ms if phase == "forward" else "",
            "warmup_ms": args.warmup_ms,
            "rep_ms": args.rep_ms,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
            "allocator_fraction": allocator_fraction,
            "status": "ok",
            "error": "",
        }
        try:
            p20, p50, p80, allocated, reserved = measure_phase(callable_, args.warmup_ms, args.rep_ms)
            row.update(
                {
                    "p20_ms": p20,
                    "p50_ms": p50,
                    "p80_ms": p80,
                    "peak_allocated_mib": allocated,
                    "peak_reserved_mib": reserved,
                }
            )
        except torch.OutOfMemoryError as exc:
            row.update(
                {
                    "status": "oom",
                    "error": str(exc),
                    "p20_ms": "",
                    "p50_ms": "",
                    "p80_ms": "",
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                }
            )
        rows.append(row)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
