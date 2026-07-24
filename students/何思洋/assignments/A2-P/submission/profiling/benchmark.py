from __future__ import annotations

import argparse
import json
import sys
import time
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
    summarize_samples,
    training_step,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end benchmark for A2-P.")
    add_common_args(parser)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train_step"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    model = build_model(args.model_size, args.context_length, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) if args.mode == "train_step" else None
    x, y = make_batch(args.model_size, args.batch_size, args.context_length, device)

    for _ in range(args.warmup):
        training_step(model, optimizer, x, y, args.mode, args.dtype, device)
        cuda_sync(device)

    reset_peak_memory(device)
    samples_ms: list[float] = []
    last_loss = None
    status = "ok"
    error = ""
    try:
        for _ in range(args.steps):
            start = time.perf_counter()
            out = training_step(model, optimizer, x, y, args.mode, args.dtype, device)
            cuda_sync(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
            if out.ndim == 0:
                last_loss = float(out.item())
    except torch.cuda.OutOfMemoryError as exc:
        status = "oom"
        error = str(exc).splitlines()[0]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except RuntimeError as exc:
        status = "error"
        error = str(exc).splitlines()[0]

    stats = summarize_samples(samples_ms) if samples_ms else {
        "mean_ms": None,
        "std_ms": None,
        "cv": None,
        "p50_ms": None,
        "min_ms": None,
        "max_ms": None,
    }
    row = {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
        "raw_timings_ms": json.dumps(samples_ms),
        **stats,
        **memory_stats(device),
        "last_loss": last_loss,
        "status": status,
        "error": error,
    }
    append_csv(args.output, [row])
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    write_json(metadata_path, collect_metadata(args, {"row": row}))
    return 0 if status in {"ok", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
