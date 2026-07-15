from __future__ import annotations

import copy
import hashlib
import json
from json import JSONDecodeError
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_json(path: str | Path) -> dict[str, Any]:
    with project_path(path).open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean_json(value), f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def append_jsonl(path: str | Path, value: Any) -> None:
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json(value), sort_keys=True, ensure_ascii=False, allow_nan=False))
        f.write("\n")


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if not dotted or any(not part for part in parts):
        raise ValueError(f"invalid dotted key: {dotted!r}")
    cur = config
    for part in parts[:-1]:
        child = cur.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {dotted!r}; {part!r} is not an object")
        cur = child
    cur[parts[-1]] = value


def apply_sets(config: dict[str, Any], sets: list[str]) -> dict[str, Any]:
    config = copy.deepcopy(config)
    for expr in sets:
        key, sep, raw = expr.partition("=")
        if not sep:
            raise ValueError(f"--set must be dotted.path=JSON, got {expr!r}")
        try:
            value = json.loads(raw)
        except JSONDecodeError:
            value = raw
        set_dotted(config, key, value)
    return config


def seed_all(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str | None) -> torch.device:
    if name in {None, "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return device


def parse_dtype(value: str | None) -> torch.dtype | None:
    if value is None or str(value).lower() in {"none", "null"}:
        return None
    names = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    key = str(value).removeprefix("torch.").lower()
    if key not in names:
        raise ValueError(f"unsupported dtype: {value!r}")
    return names[key]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with project_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
