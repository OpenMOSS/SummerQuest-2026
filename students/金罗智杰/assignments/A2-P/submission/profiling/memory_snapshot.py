from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from benchmark import MODEL_CONFIGS, execute_step, nvtx_range, synchronize
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a PyTorch CUDA memory snapshot for one Transformer step.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for name in ("vocab_size", "batch_size", "context_length", "max_entries"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.output.exists() and args.output.is_dir():
        parser.error("--output must be a snapshot file path, not a directory")
    if args.output.suffix not in {".pickle", ".pkl"}:
        parser.error("--output must end with .pickle or .pkl")
    return args


def metadata_path_for(snapshot_path: Path) -> Path:
    return snapshot_path.with_suffix(".metadata.json")


def get_memory_stats(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def write_metadata(
    *,
    args: argparse.Namespace,
    parameter_count: int,
    device: torch.device,
    status: str,
    memory_before_measurement: dict[str, int],
    memory_after_measurement: dict[str, int],
    snapshot_saved: bool,
    failure_stage: str | None = None,
) -> None:
    config = MODEL_CONFIGS[args.model_size]
    result: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "command": shlex.join(["python", *sys.argv]),
        "config": {
            "model_size": args.model_size,
            **asdict(config),
            "parameter_count": parameter_count,
            "vocab_size": args.vocab_size,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "max_entries": args.max_entries,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "memory_before_measurement": memory_before_measurement,
        "memory_after_measurement": memory_after_measurement,
        "snapshot": {
            "filename": args.output.name,
            "saved": snapshot_saved,
        },
    }
    if failure_stage is not None:
        result["failure"] = {
            "type": "torch.OutOfMemoryError",
            "stage": failure_stage,
        }

    path = metadata_path_for(args.output)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Memory profiling requires a CUDA device.")
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
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

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

    print(f"memory profile: model={args.model_size} mode={args.mode} dtype={args.dtype} batch={args.batch_size} context={args.context_length} warmup={args.warmup}")
    print(f"parameters={parameter_count:,}")

    with nvtx_range("memory/warmup", device):
        for _ in range(args.warmup):
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

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    memory_before = get_memory_stats(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    status = "success"
    failure_stage: str | None = None
    snapshot_saved = False
    torch.cuda.memory._record_memory_history(max_entries=args.max_entries)
    try:
        with nvtx_range("memory/measure", device):
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
        synchronize(device)
    except torch.OutOfMemoryError:
        status = "oom"
        failure_stage = args.mode
        print("status=oom")
    finally:
        try:
            torch.cuda.memory._dump_snapshot(str(args.output))
            snapshot_saved = True
        except Exception as snapshot_error:
            print(f"snapshot_save_error={type(snapshot_error).__name__}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    memory_after = get_memory_stats(device)
    write_metadata(
        args=args,
        parameter_count=parameter_count,
        device=device,
        status=status,
        memory_before_measurement=memory_before,
        memory_after_measurement=memory_after,
        snapshot_saved=snapshot_saved,
        failure_stage=failure_stage,
    )

    print(f"peak_allocated_mib={memory_after['peak_allocated_bytes'] / 1024**2:.1f}")
    print(f"peak_reserved_mib={memory_after['peak_reserved_bytes'] / 1024**2:.1f}")
    print(f"snapshot_saved={snapshot_saved}: {args.output}")
    print(f"saved metadata: {metadata_path_for(args.output)}")
    return 0 if status == "success" else 2


if __name__ == "__main__":
    raise SystemExit(main())
