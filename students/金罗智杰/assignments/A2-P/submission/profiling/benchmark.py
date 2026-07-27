from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import sys
import timeit
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

PHASE_NAMES = ("zero_grad_ms", "forward_ms", "backward_ms", "optimizer_ms", "total_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CS336 Transformer forward and training steps.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        required=True,
        help="Operations included in each measured step.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--nvtx-attention",
        action="store_true",
        help="Install the profiling-only attention implementation with score, softmax, and value NVTX ranges.",
    )
    parser.add_argument(
        "--nvtx-blocks",
        action="store_true",
        help="Add indexed forward and backward NVTX ranges to every TransformerBlock.",
    )

    # Optional overrides make it possible to benchmark hyperparameters outside Table 1.
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    args = parser.parse_args()

    for name in ("vocab_size", "batch_size", "context_length", "steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.learning_rate < 0:
        parser.error("--learning-rate must be non-negative")
    if args.output.exists() and args.output.is_dir():
        parser.error("--output must be a CSV file path, not a directory")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end with .csv")

    return args


def resolve_model_config(args: argparse.Namespace) -> ModelConfig:
    config = MODEL_CONFIGS[args.model_size]
    overrides = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }
    config = replace(config, **{key: value for key, value in overrides.items() if value is not None})

    if min(asdict(config).values()) <= 0:
        raise ValueError("All model dimensions and layer counts must be positive.")
    if config.d_model % config.num_heads != 0:
        raise ValueError(f"d_model ({config.d_model}) must be divisible by num_heads ({config.num_heads}).")
    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def nvtx_range(name: str, device: torch.device) -> Iterator[None]:
    if device.type == "cuda":
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def autocast_context(dtype_name: str, device: torch.device):
    if dtype_name == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def timed_phase(
    name: str,
    operation,
    device: torch.device,
    collect_timing: bool,
) -> tuple[Any, float | None]:
    if collect_timing:
        synchronize(device)
        start = timeit.default_timer()

    with nvtx_range(name, device):
        result = operation()

    if not collect_timing:
        return result, None

    synchronize(device)
    elapsed_ms = (timeit.default_timer() - start) * 1_000
    return result, elapsed_ms


def execute_step(
    *,
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    mode: str,
    dtype_name: str,
    device: torch.device,
    collect_timing: bool,
) -> dict[str, float | None]:
    timings: dict[str, float | None] = {
        "zero_grad_ms": None,
        "forward_ms": None,
        "backward_ms": None,
        "optimizer_ms": None,
        "total_ms": None,
    }

    # Clearing gradients is required before every backward pass. For
    # forward_backward it is deliberately outside the measured operation; for a
    # complete train_step it is part of the end-to-end step.
    if mode == "forward_backward":
        model.zero_grad(set_to_none=True)

    if collect_timing:
        synchronize(device)
        total_start = timeit.default_timer()

    if mode == "train_step":
        if optimizer is None:
            raise RuntimeError("train_step requires an optimizer.")
        _, timings["zero_grad_ms"] = timed_phase(
            "zero_grad",
            lambda: optimizer.zero_grad(set_to_none=True),
            device,
            collect_timing,
        )

    def run_forward():
        grad_context = torch.no_grad() if mode == "forward" else nullcontext()
        with grad_context, autocast_context(dtype_name, device):
            logits = model(input_ids)
            if mode == "forward":
                return logits, None
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
            return logits, loss

    (logits, loss), timings["forward_ms"] = timed_phase(
        "forward",
        run_forward,
        device,
        collect_timing,
    )

    if mode != "forward":
        if loss is None:
            raise RuntimeError("Backward mode did not produce a loss.")
        _, timings["backward_ms"] = timed_phase(
            "backward",
            loss.backward,
            device,
            collect_timing,
        )

    if mode == "train_step":
        if optimizer is None:
            raise RuntimeError("train_step requires an optimizer.")
        _, timings["optimizer_ms"] = timed_phase(
            "optimizer",
            optimizer.step,
            device,
            collect_timing,
        )

    synchronize(device)
    if collect_timing:
        timings["total_ms"] = (timeit.default_timer() - total_start) * 1_000

    del logits
    if loss is not None:
        del loss
    return timings


def summarize_timings(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for phase in PHASE_NAMES:
        values = [float(row[phase]) for row in rows if row[phase] is not None]
        if not values:
            continue
        mean_ms = statistics.mean(values)
        std_ms = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[phase.removesuffix("_ms")] = {
            "count": len(values),
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "cv": std_ms / mean_ms if mean_ms else 0.0,
            "min_ms": min(values),
            "max_ms": max(values),
        }
    return summary


def write_csv(
    path: Path,
    rows: list[dict[str, float | None]],
    args: argparse.Namespace,
    config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "measurement",
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "vocab_size",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "nvtx_attention",
        "nvtx_blocks",
        *PHASE_NAMES,
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, timings in enumerate(rows, start=1):
            writer.writerow(
                {
                    "measurement": index,
                    "model_size": args.model_size,
                    "mode": args.mode,
                    "dtype": args.dtype,
                    "batch_size": args.batch_size,
                    "context_length": args.context_length,
                    "vocab_size": args.vocab_size,
                    "nvtx_attention": args.nvtx_attention,
                    "nvtx_blocks": args.nvtx_blocks,
                    **asdict(config),
                    **timings,
                }
            )


def metadata_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}.metadata.json")


def get_environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
                "gpu_total_memory_bytes": properties.total_memory,
            }
        )
    return metadata


def main() -> None:
    args = parse_args()
    config = resolve_model_config(args)
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError("This benchmark is intended for CUDA GPUs; pass a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.nvtx_attention:
        from nvtx_ranges import install_attention_nvtx

        install_attention_nvtx()

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    ).to(device)
    model.train(args.mode != "forward")

    _nvtx_handles = []
    if args.nvtx_blocks:
        from nvtx_ranges import install_transformer_block_nvtx

        _nvtx_handles = install_transformer_block_nvtx(model)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate) if args.mode == "train_step" else None

    # Random data generation and host-to-device transfer are intentionally
    # outside both warm-up and measurement timing.
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

    print(f"benchmark: model={args.model_size} mode={args.mode} dtype={args.dtype} batch={args.batch_size} context={args.context_length} warmup={args.warmup} steps={args.steps}")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")

    with nvtx_range("profile/warmup", device):
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

    rows: list[dict[str, float | None]] = []
    with nvtx_range("profile/measure", device):
        for step in range(args.steps):
            with nvtx_range(f"measurement/step_{step + 1}", device):
                timings = execute_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    target_ids=target_ids,
                    mode=args.mode,
                    dtype_name=args.dtype,
                    device=device,
                    collect_timing=True,
                )
            rows.append(timings)
            print(f"measurement {step + 1}/{args.steps}: total_ms={timings['total_ms']:.3f}")

    summary = summarize_timings(rows)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    write_csv(args.output, rows, args, config)
    metadata_path = metadata_path_for(args.output)
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
            "warmup": args.warmup,
            "steps": args.steps,
            "dtype": args.dtype,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "nvtx_attention": args.nvtx_attention,
            "nvtx_blocks": args.nvtx_blocks,
        },
        "environment": get_environment_metadata(device),
        "statistics": {
            "standard_deviation": "sample",
            "phases": summary,
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
        },
        "raw_timings_csv": args.output.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("summary:")
    for phase, values in summary.items():
        print(f"  {phase}: mean_ms={values['mean_ms']:.3f} std_ms={values['std_ms']:.3f} cv={values['cv']:.4f}")
    print(f"peak_memory_allocated_mib={peak_allocated / 1024**2:.1f}")
    print(f"peak_memory_reserved_mib={peak_reserved / 1024**2:.1f}")
    print(f"saved timings: {args.output}")
    print(f"saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
