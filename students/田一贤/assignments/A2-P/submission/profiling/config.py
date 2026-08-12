"""Model configurations pinned to Table 1 of the assignment PDF."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    vocab_size: int = 10_000


MODEL_SPECS = {
    "small": ModelSpec(768, 3072, 12, 12),
    "medium": ModelSpec(1024, 4096, 24, 16),
    "large": ModelSpec(1280, 5120, 36, 20),
    "xl": ModelSpec(2560, 10240, 32, 32),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown model size: {name}") from exc
