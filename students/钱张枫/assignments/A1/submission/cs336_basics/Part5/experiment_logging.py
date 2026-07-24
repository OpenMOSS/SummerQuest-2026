from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO


class JsonlMetricLogger:
    """Append structured metric records and atomically write run metadata."""

    def __init__(self, log_dir: str | Path, *, append: bool) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.summary_path = self.log_dir / "summary.json"
        self.config_path = self.log_dir / "config.json"
        self._stream: TextIO = self.metrics_path.open("a" if append else "w", encoding="utf-8")

    def write_metric(self, record: Mapping[str, Any]) -> None:
        self._stream.write(
            json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        )
        self._stream.flush()

    def write_config(self, config: Mapping[str, Any]) -> None:
        write_json_atomic(self.config_path, config)

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        write_json_atomic(self.summary_path, summary)

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> JsonlMetricLogger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.name}.tmp")
    temporary_path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, destination)


def read_last_metric(metrics_path: str | Path) -> dict[str, Any] | None:
    path = Path(metrics_path)
    if not path.is_file():
        return None

    last_non_empty_line: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last_non_empty_line = line
    if last_non_empty_line is None:
        return None

    value = json.loads(last_non_empty_line)
    if not isinstance(value, dict):
        raise ValueError(f"last metric in {path} must be a JSON object.")
    return value
