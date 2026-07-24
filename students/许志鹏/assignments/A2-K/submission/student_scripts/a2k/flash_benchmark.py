from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import torch

from cs336_systems.a2k.attention import explicit_attention
from cs336_systems.a2k.runtime import (
    collect_run_metadata,
    configure_cuda_allocator,
    peak_memory_mib,
    require_formal_free_memory,
    reset_peak_memory,
    upsert_csv_rows,
    upsert_json_record,
)
from student_scripts.a2k.common import add_formal_runtime_arguments, stable_run_id, torch_dtype


FIELDS = [
    "run_id",
    "implementation",
    "sequence_length",
    "head_dim",
    "batch_size",
    "dtype",
    "is_causal",
    "phase",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "warmup_ms",
    "rep_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "speedup_vs_eager",
    "query_tile_size",
    "key_tile_size",
    "num_warps",
    "num_stages",
    "status",
    "error",
]


def parser() -> argparse.Namespace:
    root = argparse.ArgumentParser(description="A2-K FlashAttention performance matrix")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    add_formal_runtime_arguments(run)
    run.add_argument("--implementation", choices=("eager", "compiled", "triton"), required=True)
    run.add_argument("--sequence-length", type=int, required=True)
    run.add_argument("--head-dim", type=int, required=True)
    run.add_argument("--phase", choices=("forward", "backward", "forward_backward"), required=True)
    run.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    run.add_argument("--warmup-ms", type=int, default=100)
    run.add_argument("--rep-ms", type=int, default=300)

    matrix = sub.add_parser("matrix")
    add_formal_runtime_arguments(matrix)
    matrix.add_argument("--warmup-ms", type=int, default=100)
    matrix.add_argument("--rep-ms", type=int, default=300)
    matrix.add_argument("--dry-run", action="store_true")

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--csv", type=Path, required=True)
    return root.parse_args()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _quantiles(value: object) -> tuple[float, float, float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise TypeError(f"unexpected do_bench quantiles result: {value!r}")
    return tuple(float(item) for item in value)


def _implementation(name: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if name == "eager":
        return lambda: explicit_attention(q, k, v, is_causal=True)
    if name == "compiled":
        compiled = torch.compile(explicit_attention, fullgraph=True)
        return lambda: compiled(q, k, v, is_causal=True)
    from cs336_systems.a2k.triton_attention import FlashAttentionTriton

    return lambda: FlashAttentionTriton.apply(q, k, v, True)


def run_one(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("formal FlashAttention benchmarking requires --device cuda")
    torch.manual_seed(args.seed)
    allocator = configure_cuda_allocator(allocator_limit_mib=args.allocator_limit_mib)
    free_memory_mib = require_formal_free_memory(minimum_free_mib=args.minimum_free_mib)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch_dtype(args.dtype)
    run_id = stable_run_id("flash", args.implementation, f"seq{args.sequence_length}", f"d{args.head_dim}", args.dtype, args.phase)
    row: dict[str, object] = {
        "run_id": run_id,
        "implementation": args.implementation,
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim,
        "batch_size": 1,
        "dtype": args.dtype,
        "is_causal": True,
        "phase": args.phase,
        "p20_ms": "",
        "p50_ms": "",
        "p80_ms": "",
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "speedup_vs_eager": "",
        "query_tile_size": "",
        "key_tile_size": "",
        "num_warps": "",
        "num_stages": "",
        "status": "error",
        "error": "",
    }
    metadata = collect_run_metadata(
        allocator=allocator,
        command=["python", "-m", "student_scripts.a2k.flash_benchmark", *sys.argv[1:]],
        seed=args.seed,
        timer="triton.testing.do_bench",
        warmup={"milliseconds": args.warmup_ms},
        measurement={"milliseconds": args.rep_ms, "quantiles": [0.2, 0.5, 0.8]},
        commit=_commit(),
        tf32_enabled=False,
    )
    metadata.update({"run_id": run_id, "experiment": "flash_benchmark", "free_memory_mib_at_start": free_memory_mib})

    try:
        q = torch.randn(1, args.sequence_length, args.head_dim, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        do = torch.randn_like(q)
        forward = _implementation(args.implementation, q, k, v)
        if args.phase == "forward":
            fn = forward
        elif args.phase == "forward_backward":
            fn = lambda: torch.autograd.grad(forward(), (q, k, v), do, create_graph=False)
        else:
            output = forward()

            def fn():
                return torch.autograd.grad(output, (q, k, v), do, retain_graph=True, create_graph=False)

        reset_peak_memory()
        import triton.testing

        result = triton.testing.do_bench(fn, warmup=args.warmup_ms, rep=args.rep_ms, quantiles=[0.2, 0.5, 0.8])
        p20, p50, p80 = _quantiles(result)
        row.update({"p20_ms": p20, "p50_ms": p50, "p80_ms": p80, **peak_memory_mib(), "status": "success"})
        if args.implementation == "triton":
            from cs336_systems.a2k.triton_attention import triton_launch_config

            row.update(triton_launch_config(args.head_dim))
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


def finalize(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if "speedup_vs_eager" not in fields:
        fields.append("speedup_vs_eager")
    lookup = {
        (row["sequence_length"], row["head_dim"], row["dtype"], row["is_causal"], row["phase"]): row
        for row in rows
        if row["implementation"] == "eager" and row["status"] == "success"
    }
    for row in rows:
        key = (row["sequence_length"], row["head_dim"], row["dtype"], row["is_causal"], row["phase"])
        eager = lookup.get(key)
        if eager and row["status"] == "success" and row["implementation"] != "eager":
            row["speedup_vs_eager"] = float(eager["p50_ms"]) / float(row["p50_ms"])
    upsert_csv_rows(path, rows, key_fields=["run_id"], fieldnames=fields)
    print(f"finalized speedups in {path}")
    return 0


def _worker(args: argparse.Namespace, implementation: str, sequence_length: int, head_dim: int, phase: str, dtype: str = "bf16") -> list[str]:
    return [
        sys.executable,
        "-m",
        "student_scripts.a2k.flash_benchmark",
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
        "--implementation",
        implementation,
        "--sequence-length",
        str(sequence_length),
        "--head-dim",
        str(head_dim),
        "--phase",
        phase,
        "--dtype",
        dtype,
        "--warmup-ms",
        str(args.warmup_ms),
        "--rep-ms",
        str(args.rep_ms),
    ]


def run_matrix(args: argparse.Namespace) -> int:
    commands: list[list[str]] = []
    for sequence_length in (512, 2048, 8192):
        for head_dim in (64, 128):
            for phase in ("forward", "backward", "forward_backward"):
                for implementation in ("eager", "compiled", "triton"):
                    commands.append(_worker(args, implementation, sequence_length, head_dim, phase))
    for sequence_length in (16384,):
        for head_dim in (64, 128):
            for phase in ("forward", "backward", "forward_backward"):
                for implementation in ("eager", "triton"):
                    commands.append(_worker(args, implementation, sequence_length, head_dim, phase))
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    if not args.dry_run:
        finalize(args.output)
    return 0


def main() -> int:
    args = parser()
    if args.command == "run":
        return run_one(args)
    if args.command == "finalize":
        return finalize(args.csv)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
