"""Model construction and checkpoint helpers shared by experiment CLIs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .transformer import TransformerLM


def canonical_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize architecture fields so semantic mismatches are detectable."""

    return {
        "vocab_size": int(model_config["vocab_size"]),
        "context_length": int(model_config["context_length"]),
        "d_model": int(model_config["d_model"]),
        "num_layers": int(model_config["num_layers"]),
        "num_heads": int(model_config["num_heads"]),
        "d_ff": int(model_config["d_ff"]),
        "rope_theta": float(model_config.get("rope_theta", 10_000.0)),
        "norm_style": str(model_config.get("norm_style", "pre")),
        "position_encoding": str(model_config.get("position_encoding", "rope")),
        "ffn_type": str(model_config.get("ffn_type", "swiglu")),
        "rms_norm_eps": float(model_config.get("rms_norm_eps", 1e-5)),
    }


def checkpoint_model_config(payload: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Prefer the architecture saved with a checkpoint over an external file."""

    checkpoint_config = payload.get("config")
    if isinstance(checkpoint_config, dict) and isinstance(checkpoint_config.get("model"), dict):
        return dict(checkpoint_config["model"])
    return dict(fallback)


def build_transformer(
    model_config: dict[str, Any],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> TransformerLM:
    """Construct a :class:`TransformerLM` from a JSON model section."""

    required = [
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
    ]
    missing = [key for key in required if key not in model_config]
    if missing:
        raise KeyError(f"model configuration is missing: {', '.join(missing)}")
    normalized = canonical_model_config(model_config)
    return TransformerLM(
        vocab_size=normalized["vocab_size"],
        context_length=normalized["context_length"],
        d_model=normalized["d_model"],
        num_layers=normalized["num_layers"],
        num_heads=normalized["num_heads"],
        d_ff=normalized["d_ff"],
        rope_theta=normalized["rope_theta"],
        norm_style=normalized["norm_style"],
        position_encoding=normalized["position_encoding"],
        ffn_type=normalized["ffn_type"],
        eps=normalized["rms_norm_eps"],
        device=device,
        dtype=dtype,
    )


def parameter_count(model: torch.nn.Module) -> int:
    """Return the number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_checkpoint_payload(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load and validate a training checkpoint dictionary."""

    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only keyword.
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dictionary: {checkpoint_path}")
    return payload


def extract_model_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Accept both assignment and common checkpoint key spellings."""

    state = payload.get("model", payload.get("model_state_dict"))
    if state is None:
        # A raw state dict is also convenient for inference.
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            state = payload
        else:
            raise KeyError("checkpoint does not contain model state")
    return {key.removeprefix("_orig_mod."): value for key, value in state.items()}


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write a checkpoint to a sibling temporary file and atomically replace."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
