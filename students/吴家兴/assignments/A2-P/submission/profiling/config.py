"""Shared model and experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


VOCAB_SIZE = 10_000
ROPE_THETA = 10_000.0

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {
        "d_model": 768,
        "d_ff": 3_072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1_024,
        "d_ff": 4_096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1_280,
        "d_ff": 5_120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2_560,
        "d_ff": 10_240,
        "num_layers": 32,
        "num_heads": 32,
    },
}

VALID_MODES = ("forward", "forward_backward", "train_step")
VALID_DTYPES = ("fp32", "bf16")


@dataclass(frozen=True)
class RunConfig:
    """Configuration that fully identifies one measured workload."""

    model_size: str
    batch_size: int
    context_length: int
    mode: str
    dtype: str = "fp32"
    warmup_steps: int = 5
    measurement_steps: int = 10
    seed: int = 42

    def validate(self) -> None:
        if self.model_size not in MODEL_CONFIGS:
            raise ValueError(f"unknown model size: {self.model_size}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        if self.dtype not in VALID_DTYPES:
            raise ValueError(f"unknown dtype: {self.dtype}")
        if self.batch_size <= 0 or self.context_length <= 0:
            raise ValueError("batch size and context length must be positive")
        if self.warmup_steps < 0 or self.measurement_steps <= 0:
            raise ValueError("warmup must be non-negative and steps must be positive")

    @property
    def run_id(self) -> str:
        return (
            f"{self.model_size}-b{self.batch_size}-c{self.context_length}-"
            f"{self.mode}-{self.dtype}-w{self.warmup_steps}-n{self.measurement_steps}"
        )

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def model_kwargs(model_size: str, context_length: int) -> dict[str, int | float]:
    """Return the fixed upstream TransformerLM constructor arguments."""

    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"unknown model size: {model_size}")
    return {
        "vocab_size": VOCAB_SIZE,
        "context_length": context_length,
        **MODEL_CONFIGS[model_size],
        "rope_theta": ROPE_THETA,
    }


def residual_stream_mib(
    model_size: str,
    batch_size: int,
    context_length: int,
    element_bytes: int = 4,
) -> float:
    """Theoretical size of one [batch, sequence, d_model] residual tensor."""

    d_model = MODEL_CONFIGS[model_size]["d_model"]
    return batch_size * context_length * d_model * element_bytes / 2**20
