from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from profiling.benchmark import execute_step
from profiling.config import (
    MODEL_CONFIGS,
    autocast_context,
    build_model,
    build_optimizer,
    cuda_memory_metrics,
    model_config,
    public_command,
    public_environment,
    public_path,
    resolve_device,
    sanitize_error,
    seed_everything,
    synchronize,
)
from profiling.nvtx_ranges import instrument_attention, profile_range


def _record_history(max_entries: int) -> None:
    try:
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=max_entries,
        )
    except TypeError:
        torch.cuda.memory._record_memory_history(max_entries=max_entries)


def _snapshot_max_allocation(snapshot: dict[str, Any]) -> float | None:
    maximum = 0
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            maximum = max(maximum, int(block.get("size", 0)))
            for item in block.get("history", []):
                maximum = max(maximum, int(item.get("real_size", item.get("size", 0))))
    return maximum / (1024**2) if maximum else None


def _current_memory(device: torch.device) -> dict[str, float | None]:
    return {
        "allocated_mib": round(float(torch.cuda.memory_allocated(device)) / (1024**2), 3),
        "reserved_mib": round(float(torch.cuda.memory_reserved(device)) / (1024**2), 3),
        "active_mib": round(
            float(torch.cuda.memory_stats(device).get("active_bytes.all.current", 0)) / (1024**2),
            3,
        ),
    }


def run_memory_profile(
    *,
    model_size: str,
    mode: str,
    dtype: str,
    batch_size: int,
    context_length: int,
    warmup: int,
    warmup_mode: str,
    device_name: str,
    seed: int,
    snapshot_path: str | Path,
    output_path: str | Path,
    max_entries: int = 1_000_000,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"forward", "train_step"}:
        raise ValueError("memory mode must be forward or train_step")
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"unknown model size: {model_size}")
    device = resolve_device(device_name)
    if device.type != "cuda":
        raise RuntimeError("memory snapshots require CUDA")
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshot_path = Path(snapshot_path)
    output_path = Path(output_path)
    result: dict[str, Any] = {
        "status": "success",
        "experiment": "memory_profile",
        "config": {
            "model_size": model_size,
            "mode": mode,
            "dtype": dtype,
            "batch_size": batch_size,
            "context_length": context_length,
            "warmup": warmup,
            "warmup_mode": warmup_mode,
            "seed": seed,
        },
        "model_config": model_config(model_size, context_length),
        "command": command,
        "environment": public_environment(device),
        "snapshot_path": public_path(snapshot_path),
        "result_path": public_path(output_path),
        "memory_points": {},
        "failed_stage": None,
    }
    model = None
    optimizer = None
    history_started = False
    try:
        model = build_model(model_size, context_length, device)
        model.train(mode == "train_step")
        optimizer = build_optimizer(model) if mode == "train_step" else None
        cfg = model_config(model_size, context_length)
        tokens = torch.randint(
            0,
            cfg["vocab_size"],
            (batch_size, context_length),
            device=device,
        )
        targets = torch.randint_like(tokens, high=cfg["vocab_size"])
        result["memory_points"]["after_model_and_inputs"] = _current_memory(device)

        # The default warm-up is a forward pass so that a training OOM still has
        # a valid snapshot. The metadata makes this choice explicit.
        for _ in range(warmup):
            if warmup_mode == "same":
                execute_step(model, optimizer, tokens, targets, mode, dtype, device)
            else:
                with torch.no_grad(), autocast_context(device, dtype):
                    model(tokens)
            synchronize(device)
        result["memory_points"]["after_warmup"] = _current_memory(device)
        torch.cuda.reset_peak_memory_stats(device)

        _record_history(max_entries)
        history_started = True
        with instrument_attention():
            if mode == "forward":
                result["failed_stage"] = "forward"
                with torch.no_grad(), autocast_context(device, dtype), profile_range("forward"):
                    model(tokens)
            else:
                result["failed_stage"] = "zero_grad"
                with profile_range("optimizer/zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                result["failed_stage"] = "forward"
                with autocast_context(device, dtype), profile_range("forward"):
                    logits = model(tokens)
                result["memory_points"]["after_forward"] = _current_memory(device)
                result["failed_stage"] = "loss"
                with profile_range("loss"):
                    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                result["failed_stage"] = "backward"
                with profile_range("backward"):
                    loss.backward()
                result["memory_points"]["after_backward"] = _current_memory(device)
                result["failed_stage"] = "optimizer"
                with profile_range("optimizer"):
                    optimizer.step()
                result["memory_points"]["after_optimizer"] = _current_memory(device)
            synchronize(device)
            result["failed_stage"] = None

        result["memory"] = cuda_memory_metrics(device)
        result["peak_allocation_from_snapshot_mib"] = None
        result["status"] = "success"
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["exception_type"] = type(exc).__name__
        result["error"] = sanitize_error(exc)
        result["memory"] = cuda_memory_metrics(device)
    except Exception as exc:
        result["status"] = "failed"
        result["exception_type"] = type(exc).__name__
        result["error"] = sanitize_error(exc)
        result["memory"] = cuda_memory_metrics(device)
    finally:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if history_started:
            try:
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
            finally:
                torch.cuda.memory._record_memory_history(enabled=None)
            try:
                with snapshot_path.open("rb") as handle:
                    snapshot = pickle.load(handle)
                result["peak_allocation_from_snapshot_mib"] = _snapshot_max_allocation(snapshot)
            except Exception:
                result["peak_allocation_from_snapshot_mib"] = None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P CUDA memory history snapshot")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--warmup-mode", choices=("forward", "same"), default="forward")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_args(arguments)
    cli_arguments = list(arguments) if arguments is not None else sys.argv[1:]
    result = run_memory_profile(
        model_size=args.model_size,
        mode=args.mode,
        dtype=args.dtype,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup=args.warmup,
        warmup_mode=args.warmup_mode,
        device_name=args.device,
        seed=args.seed,
        snapshot_path=args.snapshot,
        output_path=args.output,
        max_entries=args.max_entries,
        command=public_command("profiling.memory_snapshot", cli_arguments),
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"success", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
