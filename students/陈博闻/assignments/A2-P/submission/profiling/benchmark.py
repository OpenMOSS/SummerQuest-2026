from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import einsum

# Make the script robust when run with a reused A1 virtualenv on an offline
# server. The assignment2 staff basics package must win over any installed A1
# package in that environment.
A2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A2_ROOT))
sys.path.insert(0, str(A2_ROOT / "cs336-basics"))

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW

try:
    from profiling.memory_snapshot import memory_history
    from profiling.nvtx_ranges import nvtx_range, synchronize_if_needed
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from memory_snapshot import memory_history
    from nvtx_ranges import nvtx_range, synchronize_if_needed


Mode = Literal["forward", "forward_backward", "train_step"]


MODEL_SIZES = {
    "tiny": {"d_model": 64, "d_ff": 256, "num_layers": 2, "num_heads": 4},
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    vocab_size: int
    batch_size: int
    context_length: int
    mode: Mode
    warmup: int
    steps: int
    dtype: str
    seed: int
    device: str
    learning_rate: float
    compile_model: bool
    nvtx: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Transformer benchmark for CS336 Assignment 2 profiling.")
    parser.add_argument("--model-size", choices=MODEL_SIZES.keys(), default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=Path("results/timings/benchmark.csv"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-nvtx", action="store_true")
    parser.add_argument("--no-attention-nvtx", action="store_true")
    parser.add_argument("--memory-snapshot", type=Path, default=None)
    parser.add_argument("--memory-history", action="store_true")
    return parser.parse_args()


def configure_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device.index)
        torch.set_float32_matmul_precision("high")
    return device


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "fp32" or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=DTYPES[dtype_name])


def build_model(config: BenchmarkConfig, device: torch.device) -> torch.nn.Module:
    model_kwargs = MODEL_SIZES[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        **model_kwargs,
    ).to(device)
    model.train()
    if config.compile_model:
        model = torch.compile(model)
    return model


def install_attention_nvtx(enabled: bool) -> None:
    if not enabled:
        return

    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        with nvtx_range("attention/scores", enabled):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            with nvtx_range("attention/mask", enabled):
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range("attention/softmax", enabled):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx_range("attention/value", enabled):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def run_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> float | None:
    optimizer.zero_grad(set_to_none=True)

    if config.mode == "forward":
        with torch.no_grad(), nvtx_range("forward", config.nvtx), autocast_context(device, config.dtype):
            logits = model(inputs)
        return None

    with nvtx_range("forward", config.nvtx), autocast_context(device, config.dtype):
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

    with nvtx_range("backward", config.nvtx):
        loss.backward()

    if config.mode == "train_step":
        with nvtx_range("optimizer", config.nvtx):
            optimizer.step()

    return float(loss.detach().cpu())


def step_time(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> tuple[float, float | None]:
    synchronize_if_needed(device)
    start = timeit.default_timer()
    loss = run_one_step(model, optimizer, inputs, targets, config, device)
    synchronize_if_needed(device)
    end = timeit.default_timer()
    return end - start, loss


def collect_environment(device: torch.device) -> dict[str, object]:
    env: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        env.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_gib": round(props.total_memory / (1024**3), 2),
            }
        )
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            driver_version = result.stdout.splitlines()[0].strip()
            if driver_version:
                env["driver_version"] = driver_version
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return env


def collect_memory(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "memory_allocated_bytes": torch.cuda.memory_allocated(device),
        "memory_reserved_bytes": torch.cuda.memory_reserved(device),
    }


def summarize_timings(timings: list[float]) -> dict[str, float]:
    mean = statistics.fmean(timings) if timings else float("nan")
    std = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return {
        "mean_seconds": mean,
        "std_seconds": std,
        "cv": std / mean if mean else float("nan"),
        "min_seconds": min(timings) if timings else float("nan"),
        "max_seconds": max(timings) if timings else float("nan"),
    }


def write_csv(path: Path, config: BenchmarkConfig, environment: dict[str, object], timings: list[float], losses: list[float | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_timings(timings)
    file_exists = path.exists()
    fieldnames = [
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "warmup",
        "steps",
        "seed",
        "device",
        "accelerator",
        "step_index",
        "seconds",
        "loss",
        "mean_seconds",
        "std_seconds",
        "cv",
        "min_seconds",
        "max_seconds",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for idx, seconds in enumerate(timings):
            writer.writerow(
                {
                    **{k: getattr(config, k) for k in ("model_size", "mode", "dtype", "batch_size", "context_length", "warmup", "steps", "seed")},
                    "device": config.device,
                    "accelerator": environment.get("gpu", ""),
                    "step_index": idx,
                    "seconds": seconds,
                    "loss": "" if losses[idx] is None else losses[idx],
                    **summary,
                }
            )


def write_json(
    path: Path,
    config: BenchmarkConfig,
    environment: dict[str, object],
    memory: dict[str, int],
    timings: list[float],
    losses: list[float | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "environment": environment,
        "memory": memory,
        "timings_seconds": timings,
        "losses": losses,
        "summary": summarize_timings(timings),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = configure_device(args.device)
    config = BenchmarkConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        warmup=args.warmup,
        steps=args.steps,
        dtype=args.dtype,
        seed=args.seed,
        device=str(device),
        learning_rate=args.learning_rate,
        compile_model=args.compile_model,
        nvtx=not args.no_nvtx,
    )

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    install_attention_nvtx(config.nvtx and not args.no_attention_nvtx)
    model = build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    inputs, targets = make_batch(config, device)
    environment = collect_environment(device)

    with nvtx_range("profile/warmup", config.nvtx):
        for _ in range(config.warmup):
            run_one_step(model, optimizer, inputs, targets, config, device)
        synchronize_if_needed(device)

    timings: list[float] = []
    losses: list[float | None] = []
    with memory_history(args.memory_snapshot, args.memory_history, max_entries=1_000_000):
        with nvtx_range("profile/measure", config.nvtx):
            for _ in range(config.steps):
                seconds, loss = step_time(model, optimizer, inputs, targets, config, device)
                timings.append(seconds)
                losses.append(loss)

    write_csv(args.output, config, environment, timings, losses)
    json_output = args.json_output or args.output.with_suffix(".json")
    memory = collect_memory(device)
    write_json(json_output, config, environment, memory, timings, losses)

    print(
        json.dumps(
            {"config": asdict(config), "summary": summarize_timings(timings), "memory": memory, "json_output": str(json_output)},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
