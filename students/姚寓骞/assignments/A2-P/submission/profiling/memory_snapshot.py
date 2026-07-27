"""Capture PyTorch CUDA allocator snapshots around one measured step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from profiling.benchmark import MODEL_CONFIGS, environment_metadata, execute_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("memory snapshots require a CUDA GPU")
    device = torch.device("cuda")
    from cs336_basics.model import BasicsTransformerLM

    model = BasicsTransformerLM(args.vocab_size, args.context_length, **MODEL_CONFIGS[args.model_size]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    metadata = {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "environment": environment_metadata(device),
        "command": " ".join(sys.argv),
    }
    try:
        for _ in range(args.warmup):
            execute_step(model, inputs, targets, optimizer, args.mode, args.dtype, device)
    except torch.OutOfMemoryError as error:
        stats = {
            **metadata,
            "status": "oom",
            "phase": "warmup",
            "allocated_bytes_at_oom": torch.cuda.memory_allocated(device),
            "reserved_bytes_at_oom": torch.cuda.memory_reserved(device),
            "active_peak_bytes": torch.cuda.memory_stats(device).get("active_bytes.all.peak", 0),
            "error_type": type(error).__name__,
            "error": str(error).split(" See documentation for Memory Management")[0],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    try:
        execute_step(model, inputs, targets, optimizer, args.mode, args.dtype, device)
        torch.cuda.synchronize()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.output))
        stats = {
            **metadata,
            "status": "success",
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            "active_peak_bytes": torch.cuda.memory_stats(device).get("active_bytes.all.peak", 0),
            "snapshot": str(args.output),
        }
        args.output.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
