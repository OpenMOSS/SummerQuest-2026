"""Small, dependency-free helpers for experiment configuration files."""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON object and remember the source path for diagnostics."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a JSON object: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def parse_scalar(value: str) -> Any:
    """Parse a command-line override value as JSON, then as a Python literal."""

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Return a deep copy with ``section.key=value`` overrides applied."""

    result = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override must have the form dotted.path=value, got {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError(f"override has an empty key: {override!r}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            existing = cursor.get(key)
            if existing is None:
                existing = {}
                cursor[key] = existing
            if not isinstance(existing, dict):
                raise ValueError(f"cannot descend through non-object key {key!r} in {override!r}")
            cursor = existing
        cursor[keys[-1]] = parse_scalar(raw_value)
    return result


def project_root() -> Path:
    """Return the assignment repository root."""

    return Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path | None, *, root: Path | None = None) -> Path | None:
    """Resolve relative artifact paths from the repository root."""

    if path is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (root or project_root()) / candidate
    return candidate.resolve()


def make_json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with JSON ``null`` values."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: make_json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(child) for child in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        json.dump(
            make_json_safe(value),
            output_file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        output_file.write("\n")
