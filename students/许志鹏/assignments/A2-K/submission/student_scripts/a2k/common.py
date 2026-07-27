from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def torch_dtype(name: str) -> torch.dtype:
    mapping = {"fp32": torch.float32, "bf16": torch.bfloat16}
    try:
        return mapping[name]
    except KeyError as error:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from error


def json_cell(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_run_id(*parts: object) -> str:
    return "-".join(str(part).lower().replace("_", "-") for part in parts)


def add_formal_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allocator-limit-mib", type=int, default=23552)
    parser.add_argument("--minimum-free-mib", type=int, default=22528)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
