from __future__ import annotations

import csv
import json
import re
import shlex
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    if not slug:
        raise ValueError("run name must contain at least one letter or number")
    return slug


def artifact_name(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.is_absolute():
        return path.name
    return path.as_posix()


def sanitized_command(argv: list[str] | None = None) -> str:
    """Return a reproducible command without leaking absolute local paths."""
    source = list(sys.argv if argv is None else argv)
    sanitized: list[str] = []
    for index, token in enumerate(source):
        path = Path(token)
        if index == 0:
            parts = path.parts
            if "profiling" in parts:
                profiling_index = parts.index("profiling")
                token = Path(*parts[profiling_index:]).as_posix()
            else:
                token = path.name
        elif path.is_absolute():
            token = path.name
        sanitized.append(token)
    return "python " + shlex.join(sanitized)


def append_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as file:
            existing_header = next(csv.reader(file), [])
        if existing_header != fieldnames:
            raise ValueError(f"CSV schema mismatch for {path}: expected {existing_header}, got {fieldnames}")
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(materialized)


def upsert_csv_rows(path: Path, rows: Iterable[dict[str, Any]], *, key: str = "run_id") -> None:
    materialized = list(rows)
    if not materialized:
        return
    if any(key not in row for row in materialized):
        raise ValueError(f"every row must contain the upsert key {key!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0])
    existing: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != fieldnames:
                raise ValueError(f"CSV schema mismatch for {path}: expected {reader.fieldnames}, got {fieldnames}")
            existing = list(reader)
    replacement_keys = {str(row[key]) for row in materialized}
    retained = [row for row in existing if str(row[key]) not in replacement_keys]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*retained, *materialized])


def append_json_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, list):
            raise ValueError(f"expected a JSON list in {path}")
        records = loaded
    records.append(record)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def upsert_json_record(path: Path, record: dict[str, Any], *, key: str = "run_id") -> None:
    if key not in record:
        raise ValueError(f"record must contain the upsert key {key!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text())
        if not isinstance(loaded, list):
            raise ValueError(f"expected a JSON list in {path}")
        records = [item for item in loaded if item.get(key) != record[key]]
    records.append(record)
    records.sort(key=lambda item: str(item.get(key, "")))
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
