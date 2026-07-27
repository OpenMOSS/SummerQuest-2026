from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch

from cs336_systems.a2k.attention import explicit_attention
from cs336_systems.a2k.runtime import (
    collect_run_metadata,
    configure_cuda_allocator,
    peak_memory_mib,
    require_formal_free_memory,
    reset_peak_memory,
    synchronize,
    timing_summary,
    upsert_csv_rows,
    upsert_json_record,
)
from student_scripts.a2k.common import add_formal_runtime_arguments, json_cell, stable_run_id, torch_dtype


FIELDS = [
    "run_id",
    "sequence_length",
    "head_dim",
    "batch_size",
    "dtype",
    "is_causal",
    "phase",
    "warmup_steps",
    "measurement_steps",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error",
]


def parser() -> argparse.Namespace:
    root = argparse.ArgumentParser(description="A2-K explicit PyTorch attention benchmark")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    add_formal_runtime_arguments(run)
    run.add_argument("--sequence-length", type=int, required=True)
    run.add_argument("--head-dim", type=int, required=True)
    run.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    run.add_argument("--phase", choices=("forward", "backward", "forward_backward"), required=True)
    run.add_argument("--warmup-steps", type=int, default=100)
    run.add_argument("--measurement-steps", type=int, default=300)
    run.add_argument("--batch-size", type=int, default=1)

    matrix = sub.add_parser("matrix")
    add_formal_runtime_arguments(matrix)
    matrix.add_argument("--warmup-steps", type=int, default=100)
    matrix.add_argument("--measurement-steps", type=int, default=300)
    matrix.add_argument("--dry-run", action="store_true")
    return root.parse_args()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _bench(fn: Callable[[], object], warmup_steps: int, measurement_steps: int) -> list[float]:
    for _ in range(warmup_steps):
        fn()
        synchronize()
    samples: list[float] = []
    for _ in range(measurement_steps):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return samples


def run_one(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("formal attention benchmarking requires --device cuda")
    torch.manual_seed(args.seed)
    allocator = configure_cuda_allocator(allocator_limit_mib=args.allocator_limit_mib)
    free_memory_mib = require_formal_free_memory(minimum_free_mib=args.minimum_free_mib)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch_dtype(args.dtype)
    run_id = stable_run_id("attention", f"seq{args.sequence_length}", f"d{args.head_dim}", args.dtype, args.phase)
    row: dict[str, object] = {
        "run_id": run_id,
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "is_causal": True,
        "phase": args.phase,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "p20_ms": "",
        "p50_ms": "",
        "p80_ms": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "error",
        "error": "",
    }
    metadata = collect_run_metadata(
        allocator=allocator,
        command=["python", "-m", "student_scripts.a2k.attention_baseline", *sys.argv[1:]],
        seed=args.seed,
        timer="perf_counter with CUDA synchronization",
        warmup={"steps": args.warmup_steps},
        measurement={"steps": args.measurement_steps},
        commit=_commit(),
        tf32_enabled=False,
    )
    metadata.update({"run_id": run_id, "experiment": "attention_baseline", "free_memory_mib_at_start": free_memory_mib})

    try:
        q = torch.randn(args.batch_size, args.sequence_length, args.head_dim, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        if args.phase == "forward":
            fn = lambda: explicit_attention(q, k, v, is_causal=True)
        elif args.phase == "forward_backward":
            fn = lambda: torch.autograd.grad(
                explicit_attention(q, k, v, is_causal=True).sum(), (q, k, v), create_graph=False
            )
        else:
            output = explicit_attention(q, k, v, is_causal=True)
            grad_output = torch.ones_like(output)

            def fn() -> object:
                q.grad = k.grad = v.grad = None
                return output.backward(grad_output, retain_graph=True)

        reset_peak_memory()
        samples = _bench(fn, args.warmup_steps, args.measurement_steps)
        summary = timing_summary(samples)
        row.update(
            {
                "p20_ms": summary["p20_ms"],
                "p50_ms": summary["p50_ms"],
                "p80_ms": summary["p80_ms"],
                **peak_memory_mib(),
                "status": "success",
                "samples_ms": json_cell(samples),
            }
        )
    except torch.OutOfMemoryError as error:
        row.update({"status": "oom", "error": str(error).replace("\n", " ")[:500]})
        try:
            row.update(peak_memory_mib())
        except RuntimeError:
            pass
    except Exception as error:
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"[:500]})

    metadata["result"] = {key: row[key] for key in ("status", "peak_allocated_mib", "peak_reserved_mib", "error")}
    upsert_csv_rows(args.output, [row], key_fields=["run_id"], fieldnames=FIELDS)
    upsert_json_record(args.metadata_output, metadata, key_fields=["run_id"])
    print(f"{run_id}: {row['status']}")
    return 0 if row["status"] in {"success", "oom"} else 1


def _worker(args: argparse.Namespace, sequence_length: int, head_dim: int, phase: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "student_scripts.a2k.attention_baseline",
        "run",
        "--device",
        args.device,
        "--allocator-limit-mib",
        str(args.allocator_limit_mib),
        "--minimum-free-mib",
        str(args.minimum_free_mib),
        "--seed",
        str(args.seed),
        "--output",
        str(args.output),
        "--metadata-output",
        str(args.metadata_output),
        "--sequence-length",
        str(sequence_length),
        "--head-dim",
        str(head_dim),
        "--dtype",
        "bf16",
        "--phase",
        phase,
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
    ]


def run_matrix(args: argparse.Namespace) -> int:
    for sequence_length in (512, 2048, 8192):
        for head_dim in (64, 128):
            for phase in ("forward", "backward", "forward_backward"):
                command = _worker(args, sequence_length, head_dim, phase)
                print(" ".join(command))
                if not args.dry_run:
                    subprocess.run(command, check=True)
    return 0


def main() -> int:
    args = parser()
    return run_one(args) if args.command == "run" else run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
