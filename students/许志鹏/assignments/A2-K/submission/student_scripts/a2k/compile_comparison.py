from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
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
from student_scripts.a2k.common import MODEL_CONFIGS, add_formal_runtime_arguments, stable_run_id


FIELDS = [
    "run_id",
    "target",
    "implementation",
    "shape",
    "phase",
    "dtype",
    "cold_start_ms",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "warmup_steps",
    "measurement_steps",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error",
]


def parser() -> argparse.Namespace:
    root = argparse.ArgumentParser(description="A2-K eager vs torch.compile comparison")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    add_formal_runtime_arguments(run)
    run.add_argument("--target", choices=("attention", "model"), required=True)
    run.add_argument("--implementation", choices=("eager", "compiled"), required=True)
    run.add_argument("--phase", choices=("forward", "forward_backward", "train_step"), required=True)
    run.add_argument("--sequence-length", type=int, default=512)
    run.add_argument("--head-dim", type=int, default=64)
    run.add_argument("--model-size", default="small", choices=tuple(MODEL_CONFIGS))
    run.add_argument("--warmup-steps", type=int, default=10)
    run.add_argument("--measurement-steps", type=int, default=30)
    run.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")

    matrix = sub.add_parser("matrix")
    add_formal_runtime_arguments(matrix)
    matrix.add_argument("--warmup-steps", type=int, default=10)
    matrix.add_argument("--measurement-steps", type=int, default=30)
    matrix.add_argument("--dry-run", action="store_true")
    return root.parse_args()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _timed(fn: Callable[[], object], warmup_steps: int, measurement_steps: int) -> tuple[float, list[float]]:
    synchronize()
    cold_start = time.perf_counter()
    fn()
    synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1000
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
    return cold_ms, samples


def _model_fn(model: torch.nn.Module, optimizer: torch.optim.Optimizer, inputs: torch.Tensor, targets: torch.Tensor, phase: str) -> Callable[[], object]:
    def call() -> object:
        if phase == "forward":
            return model(inputs)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        if phase == "train_step":
            optimizer.step()
        return loss

    return call


def run_one(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("formal compile comparison requires --device cuda")
    torch.manual_seed(args.seed)
    allocator = configure_cuda_allocator(allocator_limit_mib=args.allocator_limit_mib)
    free_memory_mib = require_formal_free_memory(minimum_free_mib=args.minimum_free_mib)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target_shape = f"seq{args.sequence_length}-d{args.head_dim}" if args.target == "attention" else f"{args.model_size}-ctx{args.sequence_length}"
    run_id = stable_run_id("compile", args.target, target_shape, args.implementation, args.phase, args.dtype)
    row: dict[str, object] = {
        "run_id": run_id,
        "target": args.target,
        "implementation": args.implementation,
        "shape": target_shape,
        "phase": args.phase,
        "dtype": args.dtype,
        "cold_start_ms": "",
        "p20_ms": "",
        "p50_ms": "",
        "p80_ms": "",
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "error",
        "error": "",
    }
    metadata = collect_run_metadata(
        allocator=allocator,
        command=["python", "-m", "student_scripts.a2k.compile_comparison", *sys.argv[1:]],
        seed=args.seed,
        timer="perf_counter with CUDA synchronization",
        warmup={"steps": args.warmup_steps},
        measurement={"steps": args.measurement_steps},
        commit=_commit(),
        tf32_enabled=False,
    )
    metadata.update({"run_id": run_id, "experiment": "compile_comparison", "free_memory_mib_at_start": free_memory_mib})

    try:
        if args.target == "attention":
            dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
            q = torch.randn(1, args.sequence_length, args.head_dim, device="cuda", dtype=dtype, requires_grad=True)
            k = torch.randn_like(q, requires_grad=True)
            v = torch.randn_like(q, requires_grad=True)
            function = explicit_attention
            if args.implementation == "compiled":
                function = torch.compile(function, fullgraph=True)
            if args.phase == "forward":
                fn = lambda: function(q, k, v, True)
            else:
                def fn() -> object:
                    q.grad = k.grad = v.grad = None
                    output = function(q, k, v, True)
                    loss = output.square().mean()
                    loss.backward()
                    return loss
        else:
            config = MODEL_CONFIGS[args.model_size]
            model = BasicsTransformerLM(
                vocab_size=10000,
                context_length=args.sequence_length,
                d_model=config.d_model,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
            ).cuda()
            if args.implementation == "compiled":
                model = torch.compile(model, fullgraph=False)
            optimizer = torch.optim.AdamW(model.parameters())
            inputs = torch.randint(10000, (1, args.sequence_length), device="cuda")
            targets = torch.randint(10000, (1, args.sequence_length), device="cuda")
            fn = _model_fn(model, optimizer, inputs, targets, args.phase)

        reset_peak_memory()
        cold_ms, samples = _timed(fn, args.warmup_steps, args.measurement_steps)
        summary = timing_summary(samples)
        row.update({"cold_start_ms": cold_ms, **{key: summary[key] for key in ("p20_ms", "p50_ms", "p80_ms")}, **peak_memory_mib(), "status": "success"})
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


def _worker(args: argparse.Namespace, target: str, implementation: str, phase: str, sequence_length: int, head_dim: int, model_size: str | None = None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.compile_comparison",
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
        "--target",
        target,
        "--implementation",
        implementation,
        "--phase",
        phase,
        "--sequence-length",
        str(sequence_length),
        "--head-dim",
        str(head_dim),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
    ]
    if model_size:
        command += ["--model-size", model_size]
    return command


def run_matrix(args: argparse.Namespace) -> int:
    commands: list[list[str]] = []
    for sequence_length, head_dim in ((512, 64), (2048, 128), (8192, 128)):
        for implementation in ("eager", "compiled"):
            commands.append(_worker(args, "attention", implementation, "forward", sequence_length, head_dim))
    for phase in ("forward", "forward_backward", "train_step"):
        for implementation in ("eager", "compiled"):
            commands.append(_worker(args, "model", implementation, phase, 512, 64, "small"))
    for command in commands:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    return 0


def main() -> int:
    args = parser()
    return run_one(args) if args.command == "run" else run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
