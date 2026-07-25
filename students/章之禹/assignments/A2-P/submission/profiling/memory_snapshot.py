"""Capture CUDA memory history and peak summaries for A2-P."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from profiling.common import (
    build_model,
    build_optimizer,
    dtype_from_name,
    hardware_metadata,
    make_batch,
    make_targets,
    peak_memory_mib,
    reset_peak_memory,
    resolve_device,
    run_step,
    sanitized_command,
    synchronize,
    namespace_config,
    public_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "medium", "large", "xl", "10b"), default="xl")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--mode", choices=("forward", "train_step"), default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_memory_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    if args.context_length <= 0 or args.batch_size <= 0 or args.vocab_size <= 0:
        raise ValueError("batch size, vocabulary size, and context length must be positive")
    command = sanitized_command("python profiling/memory_snapshot.py", sys.argv[1:])
    if device.type != "cuda":
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "memory history requires CUDA",
            "config": public_config(namespace_config(args)),
            "hardware": hardware_metadata(device),
            "command": command,
        }
        write_json(args.output, result)
        return result

    amp_dtype = dtype_from_name(args.dtype)
    # Keep parameters in FP32; ``--autocast`` controls temporary compute dtype.
    parameter_dtype = torch.float32
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = {
        "model_size": args.model_size,
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "warmup_steps": args.warmup,
        "measurement_steps": 1,
        "dtype": args.dtype,
        "autocast": bool(args.autocast),
        "parameter_dtype": "torch.float32",
        "compute_dtype": str(amp_dtype if args.autocast else torch.float32),
        "seed": args.seed,
    }
    try:
        model = build_model(
            args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            device=device,
            parameter_dtype=parameter_dtype,
        )
        model.train(args.mode == "train_step")
        optimizer = None if args.mode == "forward" else build_optimizer(model)
        inputs = make_batch(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device,
            seed=args.seed + 1,
        )
        targets = (
            None
            if args.mode == "forward"
            else make_targets(inputs, vocab_size=args.vocab_size, seed=args.seed + 2)
        )
    except (RuntimeError, MemoryError) as error:
        peak_allocated, peak_reserved = peak_memory_mib(device)
        result = {
            "schema_version": 1,
            "status": "failed",
            "phase": "initialization",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": command,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
        }
        write_json(args.output, result)
        return result

    def one_step() -> None:
        run_step(
            model,
            optimizer,
            inputs,
            targets,
            mode=args.mode,
            amp_dtype=amp_dtype if args.autocast else torch.float32,
            device=device,
        )

    try:
        for _ in range(args.warmup):
            one_step()
            synchronize(device)
    except (RuntimeError, MemoryError) as error:
        peak_allocated, peak_reserved = peak_memory_mib(device)
        result = {
            "schema_version": 1,
            "status": "failed",
            "phase": "warmup",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": command,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
        }
        write_json(args.output, result)
        return result
    synchronize(device)
    reset_peak_memory(device)
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    recording = False
    try:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        recording = True
    except (AttributeError, RuntimeError) as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": f"unable to enable memory history: {error}",
            "config": public_config(namespace_config(args)),
            "hardware": hardware_metadata(device),
            "command": command,
        }
        write_json(args.output, result)
        return result
    try:
        one_step()
        synchronize(device)
        torch.cuda.memory._dump_snapshot(str(args.snapshot))
    except (RuntimeError, MemoryError) as error:
        peak_allocated, peak_reserved = peak_memory_mib(device)
        snapshot_written = False
        try:
            torch.cuda.memory._dump_snapshot(str(args.snapshot))
            snapshot_written = True
        except (AttributeError, OSError, RuntimeError):
            pass
        result = {
            "schema_version": 1,
            "status": "failed",
            "phase": "measure",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": command,
            "snapshot": str(args.snapshot),
            "snapshot_written": snapshot_written,
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
        }
        write_json(args.output, result)
        return result
    finally:
        if recording:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except (AttributeError, RuntimeError):
                pass
    peak_allocated, peak_reserved = peak_memory_mib(device)
    result = {
        "schema_version": 1,
        "status": "complete",
        "config": config,
        "hardware": hardware_metadata(device),
        "command": command,
        "snapshot": str(args.snapshot),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
    }
    write_json(args.output, result)
    return result


def main() -> int:
    args = parse_args()
    result = run_memory_snapshot(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
