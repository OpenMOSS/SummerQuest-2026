from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k.checkpointing import CheckpointedTransformerLM
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
from student_scripts.a2k.common import MODEL_CONFIGS, add_formal_runtime_arguments, json_cell, stable_run_id


CHECKPOINT_FIELDS = [
    "config_id",
    "model_size",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-K activation-checkpointing benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one configuration in this process")
    add_formal_runtime_arguments(run_parser)
    run_parser.add_argument("--context-length", type=int, required=True)
    run_parser.add_argument("--checkpoint-block-size", type=int, default=0, help="0 means no checkpointing")
    run_parser.add_argument("--warmup-steps", type=int, default=3)
    run_parser.add_argument("--measurement-steps", type=int, default=5)
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--vocab-size", type=int, default=10000)

    matrix_parser = subparsers.add_parser("matrix", help="spawn the required formal matrix serially")
    add_formal_runtime_arguments(matrix_parser)
    matrix_parser.add_argument("--warmup-steps", type=int, default=3)
    matrix_parser.add_argument("--measurement-steps", type=int, default=5)
    matrix_parser.add_argument("--batch-size", type=int, default=1)
    matrix_parser.add_argument("--vocab-size", type=int, default=10000)
    matrix_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()


def run_one(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("formal checkpoint measurements require --device cuda")
    torch.manual_seed(args.seed)
    allocator = configure_cuda_allocator(allocator_limit_mib=args.allocator_limit_mib)
    free_memory_mib = require_formal_free_memory(minimum_free_mib=args.minimum_free_mib)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    block_size = args.checkpoint_block_size or None
    config_id = stable_run_id(
        "medium",
        f"ctx{args.context_length}",
        "baseline" if block_size is None else f"block{block_size}",
        "bf16",
    )
    row: dict[str, object] = {
        "config_id": config_id,
        "model_size": "medium",
        "num_layers": MODEL_CONFIGS["medium"].num_layers,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": "bf16",
        "checkpoint_block_size": "" if block_size is None else block_size,
        "nested": False,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "step_time_ms_samples": "[]",
        "step_time_ms_p50": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "error",
        "error": "",
    }
    metadata = collect_run_metadata(
        allocator=allocator,
        command=["python", "-m", "student_scripts.a2k.checkpoint_benchmark", *sys.argv[1:]],
        seed=args.seed,
        timer="perf_counter with CUDA synchronization",
        warmup={"steps": args.warmup_steps},
        measurement={"steps": args.measurement_steps},
        commit=_commit(),
        tf32_enabled=False,
    )
    metadata.update(
        {
            "run_id": config_id,
            "experiment": "checkpointing",
            "free_memory_mib_at_start": free_memory_mib,
            "config": {key: row[key] for key in CHECKPOINT_FIELDS[:10]},
        }
    )

    try:
        model_config = MODEL_CONFIGS["medium"]
        base_model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=model_config.d_model,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            d_ff=model_config.d_ff,
        ).cuda()
        model = CheckpointedTransformerLM(base_model, block_size)
        optimizer = torch.optim.AdamW(model.parameters())
        inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device="cuda")

        for _ in range(args.warmup_steps):
            _training_step(model, optimizer, inputs, targets)
            synchronize()

        reset_peak_memory()
        samples_ms: list[float] = []
        for _ in range(args.measurement_steps):
            synchronize()
            start = time.perf_counter()
            _training_step(model, optimizer, inputs, targets)
            synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000)

        summary = timing_summary(samples_ms)
        row.update(
            {
                "step_time_ms_samples": json_cell(samples_ms),
                "step_time_ms_p50": summary["p50_ms"],
                **peak_memory_mib(),
                "status": "success",
            }
        )
    except torch.OutOfMemoryError as error:
        row.update({"status": "oom", "error": str(error).replace("\n", " ")[:500]})
        try:
            row.update(peak_memory_mib())
        except RuntimeError:
            pass
    except Exception as error:  # Keep a row for formal failures instead of silently dropping it.
        row.update({"status": "error", "error": f"{type(error).__name__}: {error}"[:500]})

    metadata["result"] = {key: row[key] for key in ("status", "peak_allocated_mib", "peak_reserved_mib", "error")}
    upsert_csv_rows(args.output, [row], key_fields=["config_id"], fieldnames=CHECKPOINT_FIELDS)
    upsert_json_record(args.metadata_output, metadata, key_fields=["run_id"])
    print(f"{config_id}: {row['status']}")
    return 0 if row["status"] in {"success", "oom"} else 1


def _worker_command(args: argparse.Namespace, context_length: int, block_size: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "student_scripts.a2k.checkpoint_benchmark",
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
        "--context-length",
        str(context_length),
        "--checkpoint-block-size",
        str(block_size),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--batch-size",
        str(args.batch_size),
        "--vocab-size",
        str(args.vocab_size),
    ]


def _best_checkpoint_block(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if row["context_length"] == "1024" and row["checkpoint_block_size"] and row["status"] == "success"
    ]
    if not candidates:
        raise RuntimeError("no successful checkpointed context-1024 row is available for the 2048 boundary run")
    return int(min(candidates, key=lambda row: float(row["peak_allocated_mib"]))["checkpoint_block_size"])


def run_matrix(args: argparse.Namespace) -> int:
    initial = [(1024, block_size) for block_size in (0, 1, 2, 4, 8)]
    for context_length, block_size in initial:
        command = _worker_command(args, context_length, block_size)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)

    if args.dry_run:
        print("# after the 1024 rows, run context 2048 for baseline and the lowest-peak successful checkpoint block")
        return 0

    best_block = _best_checkpoint_block(args.output)
    for block_size in (0, best_block):
        command = _worker_command(args, 2048, block_size)
        print(" ".join(command))
        subprocess.run(command, check=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return run_one(args)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
