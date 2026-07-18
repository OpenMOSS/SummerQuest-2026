"""Shared experiment helpers used by the command-line scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: dict[str, Any], device: Any) -> Any:
    import torch

    from cs336_basics.model import TransformerLM

    model = config["model"]
    ablation = config.get("ablation", {})
    return TransformerLM(
        vocab_size=model["vocab_size"],
        context_length=model["context_length"],
        d_model=model["d_model"],
        num_layers=model["num_layers"],
        num_heads=model["num_heads"],
        d_ff=model["d_ff"],
        rope_theta=None if ablation.get("no_rope", False) else model["rope_theta"],
        norm_mode="post" if ablation.get("post_norm", False) else "pre",
        use_rmsnorm=not ablation.get("no_rmsnorm", False),
        ffn_type=ablation.get("ffn_type", "swiglu"),
        device=device,
        dtype=torch.float32,
    )


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_dtype(name: str) -> Any:
    import torch

    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")
