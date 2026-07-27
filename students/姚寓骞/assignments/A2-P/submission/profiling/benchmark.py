"""End-to-end Transformer benchmark with stable timing boundaries."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from profiling.annotated_attention import install_annotated_attention
from profiling.nvtx_ranges import nvtx_range


MODEL_CONFIGS = {
    "tiny": {"d_model": 64, "num_layers": 2, "num_heads": 4, "d_ff": 256},
    "small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072},
    "medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096},
    "large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120},
    "xl": {"d_model": 2560, "num_layers": 32, "num_heads": 32, "d_ff": 10240},
    "10b": {"d_model": 4608, "num_layers": 50, "num_heads": 36, "d_ff": 12288},
}


@dataclass(frozen=True)
class RunConfig:
    model_size: str
    batch_size: int
    context_length: int
    vocab_size: int
    mode: str
    warmup: int
    steps: int
    dtype: str
    device: str
    seed: int


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, dtype: str):
    enabled = dtype != "fp32"
    cast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype, torch.float32)
    return torch.autocast(device_type=device.type, dtype=cast_dtype, enabled=enabled)


def execute_step(model, inputs, targets, optimizer, mode: str, dtype: str, device: torch.device):
    if mode == "forward":
        with torch.no_grad(), nvtx_range("forward"), autocast_context(device, dtype):
            return model(inputs)

    optimizer.zero_grad(set_to_none=True)
    with nvtx_range("forward"), autocast_context(device, dtype):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
    with nvtx_range("backward"):
        loss.backward()
    if mode == "train_step":
        with nvtx_range("optimizer"):
            optimizer.step()
    return loss


def environment_metadata(device: torch.device) -> dict:
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        metadata["gpu"] = torch.cuda.get_device_name(device)
        metadata["gpu_capability"] = list(torch.cuda.get_device_capability(device))
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata["driver"] = query.stdout.splitlines()[0].strip()
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            metadata["driver"] = "unavailable"
        try:
            query = subprocess.run(["nsys", "--version"], check=True, capture_output=True, text=True)
            metadata["nsight_systems"] = " ".join(query.stdout.split())
        except (FileNotFoundError, subprocess.CalledProcessError):
            metadata["nsight_systems"] = "unavailable"
    return metadata


def append_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return
    row = {**result["config"], **result["summary"], "timings_ms": json.dumps(result["timings_ms"]), "command": result["command"]}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="tiny")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train_step"), default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/timings.csv"))
    parser.add_argument(
        "--cuda-profiler-api",
        action="store_true",
        help="Bracket measurement with cudaProfilerStart/Stop for nsys --capture-range=cudaProfilerApi",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.cuda_profiler_api and device.type != "cuda":
        raise ValueError("--cuda-profiler-api requires CUDA")
    install_annotated_attention()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = RunConfig(args.model_size, args.batch_size, args.context_length, args.vocab_size, args.mode, args.warmup, args.steps, args.dtype, str(device), args.seed)
    model = BasicsTransformerLM(vocab_size=args.vocab_size, context_length=args.context_length, **MODEL_CONFIGS[args.model_size]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

    with nvtx_range("profile/warmup"):
        for _ in range(args.warmup):
            execute_step(model, inputs, targets, optimizer, args.mode, args.dtype, device)
    synchronize(device)

    timings = []
    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()
    with nvtx_range("profile/measure"):
        for _ in range(args.steps):
            synchronize(device)
            start = time.perf_counter()
            execute_step(model, inputs, targets, optimizer, args.mode, args.dtype, device)
            synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()

    mean = statistics.fmean(timings)
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    result = {
        "config": asdict(config),
        "model": MODEL_CONFIGS[args.model_size],
        "environment": environment_metadata(device),
        "command": " ".join(sys.argv),
        "timings_ms": timings,
        "summary": {"mean_ms": mean, "std_ms": std, "cv": std / mean if mean else 0.0},
    }
    append_result(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
