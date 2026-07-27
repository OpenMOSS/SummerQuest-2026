from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark import MODEL_CONFIGS, execute_step
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a short PyTorch CPU/CUDA operator trace.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train_step"), default="train_step")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--wait", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--active", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in ("vocab_size", "batch_size", "context_length", "active"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("wait", "warmup"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    if args.output.exists() and args.output.is_dir():
        parser.error("--output must be a Chrome trace JSON file, not a directory")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end with .json")
    return args


def companion_path(trace_path: Path, suffix: str) -> Path:
    return trace_path.with_name(f"{trace_path.stem}.{suffix}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This trace must be captured on a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    ).to(device)
    model.train(args.mode != "forward")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate) if args.mode == "train_step" else None

    input_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )
    target_ids = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        dtype=torch.long,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    operators_path = companion_path(args.output, "operators.txt")
    trace_written = False

    def save_trace(profiler: torch.profiler.profile) -> None:
        nonlocal trace_written
        profiler.export_chrome_trace(str(args.output))
        operators_path.write_text(
            profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=50) + "\n",
            encoding="utf-8",
        )
        trace_written = True

    schedule = torch.profiler.schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    total_steps = args.wait + args.warmup + args.active
    torch.cuda.reset_peak_memory_stats(device)

    print(
        f"torch profiler: model={args.model_size} mode={args.mode} dtype={args.dtype} "
        f"batch={args.batch_size} context={args.context_length} "
        f"schedule=wait:{args.wait}/warmup:{args.warmup}/active:{args.active}"
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=save_trace,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        for step in range(total_steps):
            execute_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                target_ids=target_ids,
                mode=args.mode,
                dtype_name=args.dtype,
                device=device,
                collect_timing=False,
            )
            profiler.step()
            print(f"profiler step {step + 1}/{total_steps}")

    if not trace_written:
        raise RuntimeError("The profiler schedule completed without exporting a trace.")

    properties = torch.cuda.get_device_properties(device)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": shlex.join(["python", *sys.argv]),
        "config": {
            "model_size": args.model_size,
            **asdict(config),
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "dtype": args.dtype,
            "wait": args.wait,
            "warmup": args.warmup,
            "active": args.active,
            "seed": args.seed,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "trace": args.output.name,
        "operator_table": operators_path.name,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    metadata_path = companion_path(args.output, "metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"saved Chrome trace: {args.output}")
    print(f"saved operator table: {operators_path}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
