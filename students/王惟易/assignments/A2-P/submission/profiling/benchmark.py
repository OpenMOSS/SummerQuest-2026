import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, stdev
from timeit import default_timer
from typing import Literal
from contextlib import nullcontext

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


Mode = Literal["forward", "forward_backward", "train_step"]

Precision = Literal["fp32", "bf16"]

VOCAB_SIZE = 10_000

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def precision_context(device: torch.device, precision: Precision):
    if precision == "bf16":
        return torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
        )
    return nullcontext()


def build_model(
    model_size: str,
    context_length: int,
    device: torch.device,
) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    model.to(device)
    model.train()

    return model


def prepare_step(
    mode: Mode,
    optimizer: torch.optim.Optimizer,
) -> None:
    if mode == "forward_backward":
        optimizer.zero_grad(set_to_none=True)


def run_step(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    mode: Mode,
    precision: Precision,
) -> torch.Tensor | None:
    if mode == "forward":
        with torch.no_grad():
            with precision_context(input_ids.device, precision):
                model(input_ids)
        return None
    elif mode == "forward_backward":
        with precision_context(input_ids.device, precision):
            logits = model(input_ids)
            loss = cross_entropy(logits, targets)
        loss.backward()
        return loss.detach()
    elif mode == "train_step":
        optimizer.zero_grad(set_to_none=True)
        with precision_context(input_ids.device, precision):
            logits = model(input_ids)
            loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        return loss.detach()
    else:
        raise ValueError(f"unsupported mode: {mode}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def measure_steps(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    mode: Mode,
    precision: Precision,
    warmup_steps: int,
    measurement_steps: int,
) -> tuple[list[float], list[float], float | None, float | None]:
    device = input_ids.device
    for _ in range(warmup_steps):
        prepare_step(mode, optimizer)
        run_step(model, optimizer, input_ids, targets, mode, precision)
        synchronize(device)

    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timing_ms: list[float] = []
    losses: list[float] = []
    for _ in range(measurement_steps):
        prepare_step(mode, optimizer)
        synchronize(device)
        start = default_timer()
        loss = run_step(model, optimizer, input_ids, targets, mode, precision)
        synchronize(device)
        end = default_timer()
        timing_ms.append((end - start) * 1000)
        if loss is not None:
            losses.append(loss.item())

    if device.type == "cuda":
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)
    else:
        peak_allocated_mib = None
        peak_reserved_mib = None

    return timing_ms, losses, peak_allocated_mib, peak_reserved_mib


def summarize_timings(timing_ms: list[float]) -> dict[str, object]:
    mean_ms = mean(timing_ms)
    stdev_ms = stdev(timing_ms)
    cv = stdev_ms / mean_ms

    return {
        "samples_ms": timing_ms,
        "mean_ms": mean_ms,
        "stdev_ms": stdev_ms,
        "cv": cv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one Transformer execution mode in a fresh process.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--dtype",
        choices=(
            "fp32",
            "bf16",
        ),
        default="fp32",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def collect_environment(device: torch.device) -> dict[str, object]:
    environment: dict[str, object] = {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": device.type,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        try:
            driver_version = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                .stdout.splitlines()[0]
                .strip()
            )
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            driver_version = None
        environment.update(
            {
                "gpu": properties.name,
                "gpu_memory_mib": properties.total_memory / 1024**2,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "driver_version": driver_version,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return environment


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args.model_size, args.context_length, device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (args.batch_size, args.context_length),
        device=device,
    )

    timings_ms, losses, peak_allocated_mib, peak_reserved_mib = measure_steps(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        mode=args.mode,
        precision=args.dtype,
        warmup_steps=args.warmup,
        measurement_steps=args.steps,
    )
    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": f"uv run python {shlex.join(sys.argv)}",
        "output": args.output.as_posix(),
        "config": {
            "model_size": args.model_size,
            **MODEL_CONFIGS[args.model_size],
            "vocab_size": VOCAB_SIZE,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "dtype": args.dtype,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
        },
        "environment": collect_environment(device),
        "timing": summarize_timings(timings_ms),
        "numerics": {"losses": losses},
        "memory": {
            "peak_allocated_mib": peak_allocated_mib,
            "peak_reserved_mib": peak_reserved_mib,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
