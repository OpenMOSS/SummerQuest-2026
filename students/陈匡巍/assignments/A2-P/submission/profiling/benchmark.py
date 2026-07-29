"""Unified, synchronization-correct end-to-end benchmark."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as functional

from cs336_basics.model import BasicsTransformerLM
from profiling.common import (
    MIB,
    MODEL_CONFIGS,
    configure_gpu,
    gpu_metadata,
    summarize_timings,
    write_json,
)
from profiling.nvtx_ranges import install_attention_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--annotate-attention", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allocator = configure_gpu()
    gpu = gpu_metadata()
    if args.annotate_attention:
        install_attention_ranges()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    command = (
        "python -m profiling.benchmark "
        f"--model-size {args.model_size} --batch-size {args.batch_size} "
        f"--context-length {args.context_length} --mode {args.mode} "
        f"--warmup {args.warmup} --steps {args.steps} --dtype {args.dtype}"
    )
    row = {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "compiled": args.compile,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "raw_timings_ms": "[]",
        "mean_ms": "",
        "std_ms": "",
        "cv": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "error",
        "error_type": "",
    }

    try:
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=args.context_length,
            **MODEL_CONFIGS[args.model_size],
        ).cuda()
        if args.compile:
            model = torch.compile(model, backend="inductor", fullgraph=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tokens = torch.randint(
            0,
            10_000,
            (args.batch_size, args.context_length),
            device="cuda",
        )
        targets = torch.randint(
            0,
            10_000,
            (args.batch_size, args.context_length),
            device="cuda",
        )

        def autocast_context():
            if args.dtype == "bf16":
                return torch.autocast("cuda", dtype=torch.bfloat16)
            return nullcontext()

        def run_step() -> None:
            if args.mode == "forward":
                with torch.no_grad(), autocast_context():
                    model(tokens)
                return
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(tokens)
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
            loss.backward()
            if args.mode == "train_step":
                optimizer.step()

        for _ in range(args.warmup):
            run_step()
            torch.cuda.synchronize()

        samples: list[float] = []
        allocated_peaks: list[float] = []
        reserved_peaks: list[float] = []
        for _ in range(args.steps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
            allocated_peaks.append(torch.cuda.max_memory_allocated() / MIB)
            reserved_peaks.append(torch.cuda.max_memory_reserved() / MIB)
        summary = summarize_timings(samples)
        row.update(
            {
                "raw_timings_ms": json.dumps(samples),
                **summary,
                "peak_allocated_mib": max(allocated_peaks),
                "peak_reserved_mib": max(reserved_peaks),
                "status": "ok",
            }
        )
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        row["error_type"] = "OutOfMemoryError"
    except Exception as error:
        row["status"] = "error"
        row["error_type"] = type(error).__name__

    payload = {
        "row": row,
        "command": command,
        "allocator": allocator,
        "gpu": gpu,
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "timer": "time.perf_counter with pre/post CUDA synchronization",
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if row["status"] in {"ok", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
