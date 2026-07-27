from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass(frozen=True)
class RunConfig:
    model_size: str
    batch_size: int
    context_length: int
    mode: str
    dtype: str
    seed: int
    vocab_size: int = 10_000
    learning_rate: float = 1e-3

    def validate(self) -> None:
        if self.model_size not in MODEL_CONFIGS:
            raise ValueError(f"unknown model size: {self.model_size}")
        if self.mode not in {"forward", "forward_backward", "train_step"}:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.dtype not in {"fp32", "bf16"}:
            raise ValueError(f"unknown dtype: {self.dtype}")
        if self.batch_size <= 0 or self.context_length <= 0:
            raise ValueError("batch size and context length must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal A2 measurements")
    return torch.device("cuda:0")


def autocast_context(dtype: str):
    if dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_model(config: RunConfig, device: torch.device) -> BasicsTransformerLM:
    config.validate()
    kwargs = MODEL_CONFIGS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        **kwargs,
    )
    return model.to(device)


def make_batch(config: RunConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed + 17)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.context_length + 1),
        device=device,
        generator=generator,
    )
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))


def run_model_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: RunConfig,
    optimizer: torch.optim.Optimizer | None,
) -> float | None:
    if config.mode == "forward":
        with torch.no_grad(), torch.profiler.record_function("forward"), autocast_context(config.dtype):
            model(inputs)
        return None

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)

    with torch.profiler.record_function("forward"), autocast_context(config.dtype):
        logits = model(inputs)
        loss = language_model_loss(logits, targets)
    with torch.profiler.record_function("backward"):
        loss.backward()
    if config.mode == "train_step":
        if optimizer is None:
            raise ValueError("train_step requires an optimizer")
        with torch.profiler.record_function("optimizer"):
            optimizer.step()
    return float(loss.detach())


def build_optimizer(model: torch.nn.Module, config: RunConfig) -> torch.optim.Optimizer | None:
    if config.mode != "train_step":
        return None
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def synchronize() -> None:
    torch.cuda.synchronize()


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def command_string() -> str:
    executable = Path(sys.argv[0])
    try:
        entrypoint = executable.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        entrypoint = executable.name
    return "python " + " ".join([entrypoint, *sys.argv[1:]])


def _nvidia_smi_field(query: str) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", "0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip().splitlines()[0]


def public_environment() -> dict[str, Any]:
    device = require_cuda()
    properties = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib": round(properties.total_memory / 2**20, 1),
        "driver_version": _nvidia_smi_field("driver_version"),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def config_dict(config: RunConfig) -> dict[str, Any]:
    result = asdict(config)
    result.update(MODEL_CONFIGS[config.model_size])
    return result


def safe_write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def coefficient_of_variation(values: list[float]) -> float:
    if not values:
        return math.nan
    mean = float(np.mean(values))
    if mean == 0:
        return 0.0
    return float(np.std(values, ddof=1) / mean) if len(values) > 1 else 0.0
