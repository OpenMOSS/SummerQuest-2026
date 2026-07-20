"""Reusable helpers for deterministic language-model training scripts."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import make_json_safe
from .losses import cross_entropy


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU torch, and all visible CUDA devices."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve ``auto`` and validate explicitly requested accelerators."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map a configuration string to a supported parameter dtype."""

    normalized = name.lower()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype {name!r}; choose float32, bfloat16, or float16")
    dtype = aliases[normalized]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 training on CPU is unsupported; use float32 or bfloat16")
    if device.type == "cuda" and dtype == torch.bfloat16:
        with torch.cuda.device(device):
            bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported:
            raise ValueError("this CUDA device does not support bfloat16; set dtype=float16 or float32")
    return dtype


def autocast_context(device: torch.device, dtype: torch.dtype, enabled: bool):
    """Return an autocast context only when it is useful and supported."""

    if not enabled or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_token_array(path: str | Path) -> np.ndarray:
    """Memory-map a one-dimensional ``.npy`` token array."""

    array = np.load(Path(path), mmap_mode="r")
    if array.ndim != 1:
        raise ValueError(f"token array must be one-dimensional, got {array.shape} from {path}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"token array must contain integer IDs, got {array.dtype} from {path}")
    return array


def sample_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample a deterministic random next-token batch using a supplied RNG."""

    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more than context_length tokens")
    starts = rng.integers(0, len(dataset) - context_length, size=batch_size)
    indices = starts[:, None] + np.arange(context_length + 1)[None, :]
    sampled = np.asarray(dataset[indices])
    tensor = torch.as_tensor(sampled, dtype=torch.long, device=device)
    return tensor[:, :-1], tensor[:, 1:]


@torch.inference_mode()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: torch.device,
    rng: np.random.Generator,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> float:
    """Estimate average per-token cross-entropy over random validation batches."""

    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for _ in range(num_batches):
            inputs, targets = sample_batch(dataset, batch_size, context_length, device, rng)
            with autocast_context(device, amp_dtype, amp_enabled):
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
            losses.append(float(loss.detach().cpu()))
    finally:
        model.train(was_training)
    return float(np.mean(losses))


def global_gradient_norm(parameters: Iterator[torch.nn.Parameter] | Any) -> float:
    """Compute the shared L2 norm of all available gradients."""

    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().pow(2).sum()
        squared = value if squared is None else squared + value.to(squared.device)
    return 0.0 if squared is None else float(torch.sqrt(squared).cpu())


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one JSON object and flush it immediately."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(
                make_json_safe(record),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        output_file.flush()


class ThroughputMeter:
    """Track average processed-token throughput from construction time."""

    def __init__(self, elapsed_offset: float = 0.0) -> None:
        self.started_at = time.perf_counter()
        self.elapsed_offset = float(elapsed_offset)

    def session_elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def elapsed(self) -> float:
        return self.elapsed_offset + self.session_elapsed()

    def tokens_per_second(self, processed_tokens: int) -> float:
        elapsed = self.session_elapsed()
        return processed_tokens / elapsed if elapsed > 0 else math.inf
