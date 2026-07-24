from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


@dataclass
class BenchmarkRunConfig:
    model_size: str
    batch_size: int
    context_length: int
    mode: str
    warmup: int
    steps: int
    dtype: str
    seed: int
    vocab_size: int = 10_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    torch_compile: bool = False
    enable_nvtx: bool = False


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This profiling suite requires a CUDA-capable GPU.")
    return torch.device("cuda")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_dtype(name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def autocast_context(dtype_name: str):
    if dtype_name == "fp32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=parse_dtype(dtype_name))


def ensure_parent_dir(path: str | os.PathLike[str]) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def to_repo_relative(path: str | os.PathLike[str]) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root()))
    except ValueError:
        return str(Path(path))


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def environment_metadata() -> dict[str, Any]:
    device = require_cuda()
    props = torch.cuda.get_device_properties(device)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": props.name,
        "gpu_count": torch.cuda.device_count(),
        "total_memory_bytes": props.total_memory,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": get_git_commit(),
    }


def build_model(
    *,
    model_size: str,
    context_length: int,
    vocab_size: int = 10_000,
    torch_compile: bool = False,
) -> BasicsTransformerLM:
    device = require_cuda()
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device=device)
    if torch_compile:
        model = torch.compile(model)  # type: ignore[assignment]
    return model


def make_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def random_batch(batch_size: int, context_length: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = require_cuda()
    inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)
    return inputs, targets


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    return cross_entropy(flat_logits, flat_targets)


def synchronize() -> None:
    torch.cuda.synchronize()


def timer() -> float:
    return time.perf_counter()


def summarize_timings(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    arr = np.asarray(values_ms, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    cv = float(std / mean) if mean else 0.0
    return {
        "mean_ms": mean,
        "std_ms": std,
        "cv": cv,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    return value


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def config_metadata(config: BenchmarkRunConfig) -> dict[str, Any]:
    model_config = dict(MODEL_CONFIGS[config.model_size])
    return {
        "run_config": asdict(config),
        "model_config": model_config,
        "command": " ".join(sys.argv),
        "cwd": to_repo_relative(Path.cwd()),
        "environment": environment_metadata(),
    }
