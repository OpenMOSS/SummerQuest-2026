from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CS336_BASICS = ROOT / "cs336-basics"
if str(CS336_BASICS) not in sys.path:
    sys.path.insert(0, str(CS336_BASICS))

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "small": ModelConfig("small", 10000, 768, 12, 12, 3072),
    "medium": ModelConfig("medium", 10000, 1024, 24, 16, 4096),
    "large": ModelConfig("large", 10000, 1280, 36, 20, 5120),
    "xl": ModelConfig("xl", 10000, 1600, 48, 25, 6400),
}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def build_model(model_size: str, context_length: int, device: torch.device) -> BasicsTransformerLM:
    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=10000.0,
    )
    return model.to(device)


def make_batch(model_size: str, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vocab_size = MODEL_CONFIGS[model_size].vocab_size
    tokens = torch.randint(0, vocab_size, (batch_size, context_length + 1), device=device)
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, -2).float(), labels.flatten())


@contextmanager
def autocast_for(dtype: str, device: torch.device):
    enabled = dtype == "bf16" and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        yield


def training_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, x: torch.Tensor, y: torch.Tensor, mode: str, dtype: str, device: torch.device) -> torch.Tensor:
    if mode == "forward":
        model.eval()
        with torch.no_grad(), autocast_for(dtype, device):
            return model(x)
    if mode == "forward_backward":
        model.train()
        model.zero_grad(set_to_none=True)
        with autocast_for(dtype, device):
            loss = lm_loss(model(x), y)
        loss.backward()
        return loss.detach()
    if mode == "train_step":
        if optimizer is None:
            raise ValueError("train_step requires an optimizer")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_for(dtype, device):
            loss = lm_loss(model(x), y)
        loss.backward()
        optimizer.step()
        return loss.detach()
    raise ValueError(f"unknown mode: {mode}")


def summarize_samples(samples_ms: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples_ms)
    std = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "mean_ms": mean,
        "std_ms": std,
        "cv": std / mean if mean else 0.0,
        "p50_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def collect_metadata(args: argparse.Namespace | dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    metadata: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in args_dict.items()},
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata["gpu"] = {
            "name": props.name,
            "total_memory_mib": props.total_memory / 1024**2,
        }
        metadata["nvidia_smi"] = query_nvidia_smi()
    if extra:
        metadata.update(extra)
    return metadata


def query_nvidia_smi() -> dict[str, str] | None:
    query = "name,memory.total,memory.free,driver_version,power.limit,pstate"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first = result.stdout.strip().splitlines()[0]
    keys = ["name", "memory_total", "memory_free", "driver_version", "power_limit", "pstate"]
    return dict(zip(keys, [part.strip() for part in first.split(",")], strict=False))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]
