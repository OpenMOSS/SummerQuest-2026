from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import time
from typing import Any

import torch
from tqdm.auto import tqdm

from profiling.benchmark import run_single_step
from profiling.common import (
    BenchmarkRunConfig,
    build_model,
    config_metadata,
    make_optimizer,
    random_batch,
    require_cuda,
    seed_everything,
    to_repo_relative,
    write_json,
)
from profiling.nvtx_ranges import install_attention_nvtx, nvtx_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture PyTorch CUDA memory snapshots for benchmark runs.")
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl", "10B"], default="xl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--snapshot-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument(
        "--max-entries",
        type=int,
        default=1_000,
        help="Maximum CUDA memory history entries to retain; larger values can make _snapshot() very slow.",
    )
    parser.add_argument(
        "--snapshot-dump-mode",
        choices=["builtin", "split", "skip"],
        default="builtin",
        help="Dump with PyTorch's helper, split collection/write timing, or skip the pickle dump.",
    )
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def memory_summary_dict() -> dict[str, Any]:
    stats = torch.cuda.memory_stats()
    return {
        "active_peak_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "allocated_peak_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
        "reserved_peak_bytes": int(stats.get("reserved_bytes.all.peak", 0)),
        "requested_peak_bytes": int(stats.get("requested_bytes.all.peak", 0)),
        "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
        "num_ooms": int(stats.get("num_ooms", 0)),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def synchronize_with_timing() -> float:
    start = time.perf_counter()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def snapshot_stats(snapshot: dict[str, Any]) -> dict[str, int]:
    segments = snapshot.get("segments", [])
    device_traces = snapshot.get("device_traces", [])
    return {
        "segments": len(segments),
        "blocks": sum(len(segment.get("blocks", [])) for segment in segments),
        "device_traces": len(device_traces),
        "trace_entries": sum(len(trace) for trace in device_traces),
    }


def main() -> None:
    args = parse_args()
    require_cuda()

    config = BenchmarkRunConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        warmup=args.warmup,
        steps=1,
        dtype=args.dtype,
        seed=args.seed,
        vocab_size=args.vocab_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        torch_compile=args.torch_compile,
        enable_nvtx=args.enable_nvtx,
    )

    seed_everything(config.seed)
    if config.enable_nvtx:
        install_attention_nvtx()

    model = build_model(
        model_size=config.model_size,
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        torch_compile=config.torch_compile,
    )

    optimizer = make_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
    )

    inputs, targets = random_batch(config.batch_size, config.context_length, config.vocab_size)

    if config.mode == "forward":
        model.eval()
    else:
        model.train()

    show_progress = not args.disable_progress
    with nvtx_range("profile/warmup"):
        for _ in tqdm(
            range(config.warmup),
            desc=f"mem-warmup:{config.model_size}:{config.mode}:{config.dtype}:ctx{config.context_length}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress or config.warmup == 0,
        ):
            run_single_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                mode=config.mode,
                dtype_name=config.dtype,
            )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.memory._record_memory_history(max_entries=args.max_entries)

    try:
        with nvtx_range("profile/measure"):
            for _ in tqdm(
                range(1),
                desc=f"mem-measure:{config.model_size}:{config.mode}:{config.dtype}:ctx{config.context_length}",
                leave=False,
                dynamic_ncols=True,
                disable=not show_progress,
            ):
                timing = run_single_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    mode=config.mode,
                    dtype_name=config.dtype,
                )
        synchronize_seconds = synchronize_with_timing()
        snapshot_path = Path(args.snapshot_output)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_debug_stats = None
        snapshot_collect_seconds = None
        snapshot_pickle_seconds = None
        if args.snapshot_dump_mode == "skip":
            pass
        elif args.snapshot_dump_mode == "builtin":
            snapshot_collect_start = time.perf_counter()
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            snapshot_collect_seconds = time.perf_counter() - snapshot_collect_start
        else:
            snapshot_collect_start = time.perf_counter()
            snapshot = torch.cuda.memory._snapshot()
            snapshot_collect_seconds = time.perf_counter() - snapshot_collect_start
            snapshot_debug_stats = snapshot_stats(snapshot)
            snapshot_pickle_start = time.perf_counter()
            with snapshot_path.open("wb") as f:
                pickle.dump(snapshot, f, protocol=pickle.HIGHEST_PROTOCOL)
            snapshot_pickle_seconds = time.perf_counter() - snapshot_pickle_start
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    result = config_metadata(config)
    result.update(
        {
            "artifact_type": "memory_snapshot",
            "snapshot_path": to_repo_relative(args.snapshot_output),
            "summary_path": to_repo_relative(args.summary_output),
            "timing": timing,
            "memory": memory_summary_dict(),
            "snapshot_dump": {
                "mode": args.snapshot_dump_mode,
                "synchronize_seconds": synchronize_seconds,
                "snapshot_seconds": snapshot_collect_seconds,
                "pickle_seconds": snapshot_pickle_seconds,
                "stats": snapshot_debug_stats,
            },
        }
    )
    write_json(args.summary_output, result)
    if args.snapshot_dump_mode == "skip":
        print(f"Skipped memory snapshot dump for {args.snapshot_output}")
    else:
        print(f"Saved memory snapshot to {args.snapshot_output}")
    print(f"Saved memory summary to {args.summary_output}")


if __name__ == "__main__":
    main()
