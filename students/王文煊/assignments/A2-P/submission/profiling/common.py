"""Shared helpers for A2-P profiling scripts."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import torch

MODEL_CONFIGS = {
    # size: (d_model, d_ff, num_layers, num_heads) -- PDF Table 1
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
}
VOCAB_SIZE = 10000


def build_model(model_size: str, context_length: int):
    from cs336_basics.model import BasicsTransformerLM

    d_model, d_ff, num_layers, num_heads = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    return model


def make_batch(batch_size: int, context_length: int, device: str, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), generator=g)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), generator=g)
    return x.to(device), y.to(device)


def autocast_ctx(dtype: str, device: str = "cuda"):
    if dtype == "bf16":
        return torch.autocast(device_type=device, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def collect_metadata(command: str, config: dict, results_path: str) -> dict:
    """Public-safe metadata: no hostname / username / internal paths."""
    p = torch.cuda.get_device_properties(0)
    return {
        "command": command,
        "config": config,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gib": round(p.total_memory / 2**30, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results_path": results_path,
    }


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
