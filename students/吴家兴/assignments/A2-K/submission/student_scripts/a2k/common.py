"""Shared setup and lightweight-result I/O for the A2-K scripts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import cs336_systems


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CS336_PACKAGE = SUBMISSION_ROOT / "cs336_systems"
if str(LOCAL_CS336_PACKAGE) not in cs336_systems.__path__:
    cs336_systems.__path__.append(str(LOCAL_CS336_PACKAGE))

from cs336_systems.a2k.runtime import (  # noqa: E402
    AllocatorConfig,
    configure_allocator,
    nvidia_smi_metadata,
    require_formal_free_memory,
    seed_everything,
    software_metadata,
)


STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"


@dataclass(frozen=True)
class FormalRun:
    """Process-local formal-run facts captured before tensor allocation."""

    allocator: AllocatorConfig
    free_memory_mib_at_start: float
    seed: int
    tf32_enabled: bool


def configure_formal_run(
    *,
    seed: int,
    tf32_enabled: bool,
) -> FormalRun:
    """Apply the allocator guard, verify free memory, and fix random seeds."""

    allocator = configure_allocator()
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    torch.backends.cudnn.allow_tf32 = tf32_enabled
    free_mib = require_formal_free_memory()
    seed_everything(seed)
    return FormalRun(
        allocator=allocator,
        free_memory_mib_at_start=free_mib,
        seed=seed,
        tf32_enabled=tf32_enabled,
    )


def public_run_record(
    *,
    run: FormalRun,
    experiment: str,
    command: str,
    timer: str,
    warmup: Mapping[str, Any],
    measurement: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one public-safe run metadata record."""

    record: dict[str, Any] = {
        "experiment": experiment,
        "starter_commit": STARTER_COMMIT,
        "seed": run.seed,
        "command": command,
        "hardware": {
            **nvidia_smi_metadata(),
            "free_memory_mib_at_start": run.free_memory_mib_at_start,
        },
        "software": software_metadata(),
        "allocator": run.allocator.as_public_dict(),
        "hard_limit_mib": 24 * 1024,
        "tf32_enabled": run.tf32_enabled,
        "timer": timer,
        "warmup": dict(warmup),
        "measurement": dict(measurement),
    }
    if extra:
        record["extra"] = dict(extra)
    return record


def _atomic_write(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_json(destination: str | Path, payload: Any) -> None:
    path = Path(destination)
    _atomic_write(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def upsert_json_record(
    destination: str | Path,
    record: Mapping[str, Any],
    *,
    key_fields: Sequence[str],
) -> None:
    path = Path(destination)
    existing: list[dict[str, Any]] = []
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError("metadata record file must contain a JSON list")
        existing = loaded
    key = tuple(record[field] for field in key_fields)
    retained = [
        item
        for item in existing
        if tuple(item.get(field) for field in key_fields) != key
    ]
    retained.append(dict(record))
    retained.sort(
        key=lambda item: tuple(
            str(item.get(field, "")) for field in key_fields
        )
    )
    write_json(path, retained)


def upsert_csv_rows(
    destination: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    fieldnames: Sequence[str],
) -> None:
    """Atomically replace rows sharing ``key_fields`` and keep stable order."""

    path = Path(destination)
    incoming = [dict(row) for row in rows]
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    incoming_keys = {
        tuple(str(row.get(field, "")) for field in key_fields)
        for row in incoming
    }
    merged: list[dict[str, Any]] = [
        row
        for row in existing
        if tuple(str(row.get(field, "")) for field in key_fields)
        not in incoming_keys
    ]
    merged.extend(incoming)
    merged.sort(
        key=lambda row: tuple(
            str(row.get(field, "")) for field in key_fields
        )
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(merged)
        temporary = Path(handle.name)
    os.replace(temporary, path)
