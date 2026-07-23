from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config, resolving optional base_config and section overrides."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_path = raw.pop("base_config", None)
    if base_path is None:
        return raw
    config = load_config((path.parent / base_path).resolve())
    for section in ("model", "optimizer", "training"):
        overrides = raw.pop(f"{section}_overrides", None)
        if overrides:
            config.setdefault(section, {}).update(overrides)
    config.update(raw)
    return config
