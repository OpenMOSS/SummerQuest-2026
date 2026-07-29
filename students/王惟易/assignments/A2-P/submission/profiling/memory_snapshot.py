import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from profiling.benchmark import (
    MODEL_CONFIGS,
    VOCAB_SIZE,
    build_model,
    collect_environment,
    run_step,
    synchronize,
)
from cs336_basics.optimizer import AdamW


MIB = 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one post-warmup CUDA memory snapshot and lightweight metadata.",
    )
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--snapshot-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def collect_memory_stats(device: torch.device) -> dict[str, float]:
    stats = torch.cuda.memory_stats(device)
    return {
        "current_active_mib": stats["active_bytes.all.current"] / MIB,
        "peak_active_mib": stats["active_bytes.all.peak"] / MIB,
        "current_allocated_mib": stats["allocated_bytes.all.current"] / MIB,
        "peak_allocated_mib": stats["allocated_bytes.all.peak"] / MIB,
        "current_reserved_mib": stats["reserved_bytes.all.current"] / MIB,
        "peak_reserved_mib": stats["reserved_bytes.all.peak"] / MIB,
    }


def main() -> None:
    args = parse_args()
    args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory profiling")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    status = "running"
    stage = "setup"
    failure_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    snapshot_error: str | None = None
    snapshot_written = False
    snapshot_kind: str | None = None
    history_enabled = False
    peak_reset_after_warmup = False
    measurement_started = False
    loss_value: float | None = None

    torch.cuda.reset_peak_memory_stats(device)

    try:
        stage = "build_model"
        model = build_model(args.model_size, args.context_length, device)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        stage = "build_inputs"
        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (args.batch_size, args.context_length),
            device=device,
        )
        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (args.batch_size, args.context_length),
            device=device,
        )

        stage = "warmup"
        for _ in range(args.warmup):
            run_step(
                model,
                optimizer,
                input_ids,
                targets,
                mode=args.mode,
                precision=args.dtype,
            )
            synchronize(device)

        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        peak_reset_after_warmup = True

        stage = "record_memory_history"
        torch.cuda.memory._record_memory_history(max_entries=args.max_entries)
        history_enabled = True

        stage = "measurement"
        measurement_started = True
        loss = run_step(
            model,
            optimizer,
            input_ids,
            targets,
            mode=args.mode,
            precision=args.dtype,
        )
        synchronize(device)
        if loss is not None:
            loss_value = loss.item()

        stage = "dump_snapshot"
        torch.cuda.memory._dump_snapshot(str(args.snapshot_output))
        snapshot_written = True
        snapshot_kind = "complete"
        status = "ok"
        stage = "complete"
    except torch.OutOfMemoryError as error:
        status = "oom"
        failure_stage = stage
        error_type = type(error).__name__
        error_message = str(error)
        if history_enabled:
            try:
                torch.cuda.memory._dump_snapshot(str(args.snapshot_output))
                snapshot_written = True
                snapshot_kind = "partial_oom"
            except Exception as dump_error:
                snapshot_error = f"{type(dump_error).__name__}: {dump_error}"
    finally:
        if history_enabled:
            torch.cuda.memory._record_memory_history(enabled=None)

        memory = collect_memory_stats(device)
        result = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "command": f"uv run python {shlex.join(sys.argv)}",
            "status": status,
            "stage": stage,
            "failure_stage": failure_stage,
            "error_type": error_type,
            "error_message": error_message,
            "config": {
                "model_size": args.model_size,
                **MODEL_CONFIGS[args.model_size],
                "vocab_size": VOCAB_SIZE,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "mode": args.mode,
                "dtype": args.dtype,
                "warmup_steps": args.warmup,
                "measurement_steps": 1,
                "max_history_entries": args.max_entries,
                "seed": args.seed,
                "learning_rate": args.learning_rate,
            },
            "environment": collect_environment(device),
            "measurement": {
                "started": measurement_started,
                "loss": loss_value,
                "peak_scope": ("measurement" if peak_reset_after_warmup else "process_until_failure"),
            },
            "memory": memory,
            "snapshot": {
                "output": args.snapshot_output.as_posix(),
                "written": snapshot_written,
                "kind": snapshot_kind,
                "error": snapshot_error,
            },
        }
        args.metadata_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if status == "oom":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
