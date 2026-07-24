from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.common import (
    add_common_args,
    append_csv,
    build_model,
    collect_metadata,
    cuda_sync,
    make_batch,
    memory_stats,
    reset_peak_memory,
    resolve_device,
    set_seed,
    training_step,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Memory profiling for A2-P.")
    add_common_args(parser)
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--snapshot-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    model = build_model(args.model_size, args.context_length, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) if args.mode == "train_step" else None
    x, y = make_batch(args.model_size, args.batch_size, args.context_length, device)

    status = "ok"
    error = ""
    snapshot_written = False
    try:
        for _ in range(args.warmup):
            training_step(model, optimizer, x, y, args.mode, args.dtype, device)
            cuda_sync(device)
        reset_peak_memory(device)
        if device.type == "cuda" and args.snapshot_output is not None:
            torch.cuda.memory._record_memory_history(max_entries=100000)
        training_step(model, optimizer, x, y, args.mode, args.dtype, device)
        cuda_sync(device)
        if device.type == "cuda" and args.snapshot_output is not None:
            args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(args.snapshot_output))
            snapshot_written = True
            torch.cuda.memory._record_memory_history(enabled=None)
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc).splitlines()[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    row = {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "seed": args.seed,
        **memory_stats(device),
        "snapshot_file": args.snapshot_output.name if snapshot_written and args.snapshot_output else "",
        "status": status,
        "error": error,
    }
    append_csv(args.output, [row])
    write_json(args.metadata_output or args.output.with_suffix(".metadata.json"), collect_metadata(args, {"row": row}))
    return 0 if status in {"ok", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
