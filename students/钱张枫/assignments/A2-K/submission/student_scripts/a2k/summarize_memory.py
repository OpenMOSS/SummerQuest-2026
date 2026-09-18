"""Aggregate observed A2-K allocator peaks into ``memory_evidence.json``.

This script reads lightweight CSV/JSON result artifacts only.  It does not
allocate CUDA tensors and does not manufacture a zero-valued memory result when
no actual measurement is available.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

try:  # Support `python -m` and direct execution.
    from .common import ALLOCATOR_LIMIT_MIB, HARD_MEMORY_LIMIT_MIB, default_output_dir, stderr, write_json
except ImportError:  # pragma: no cover - direct-script fallback.
    from common import ALLOCATOR_LIMIT_MIB, HARD_MEMORY_LIMIT_MIB, default_output_dir, stderr, write_json  # type: ignore[no-redef]


KNOWN_RESULT_FILES: tuple[str, ...] = (
    "memory_evidence.json",
    "attention_baseline.csv",
    "compile_comparison.csv",
    "flash_benchmark.csv",
    "checkpointing.csv",
    "correctness.json",
)


@dataclass(frozen=True)
class MemoryObservation:
    source: str
    script: str
    status: str
    formal: bool | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    allocator_fraction: float | None
    allocator_limit_mib: float | None

    def as_json(self) -> dict[str, object]:
        return {
            "source": self.source,
            "script": self.script,
            "status": self.status,
            "formal": self.formal,
            "peak_allocated_mib": round(self.peak_allocated_mib, 3) if self.peak_allocated_mib is not None else None,
            "peak_reserved_mib": round(self.peak_reserved_mib, 3) if self.peak_reserved_mib is not None else None,
            "allocator_fraction": self.allocator_fraction,
            "allocator_limit_mib": self.allocator_limit_mib,
        }


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _boolean(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _string(value: object, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _observation_from_mapping(source: str, mapping: Mapping[str, object], *, default_script: str) -> MemoryObservation:
    return MemoryObservation(
        source=source,
        script=_string(mapping.get("script"), default_script),
        status=_string(mapping.get("status"), "unknown"),
        formal=_boolean(mapping.get("formal")),
        peak_allocated_mib=_number(mapping.get("peak_allocated_mib")),
        peak_reserved_mib=_number(mapping.get("peak_reserved_mib")),
        allocator_fraction=_number(mapping.get("allocator_fraction")),
        allocator_limit_mib=_number(mapping.get("allocator_limit_mib")),
    )


def _read_json_observations(path: Path) -> list[MemoryObservation]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse JSON input {path.name!r}.") from error
    if not isinstance(content, Mapping):
        raise ValueError(f"JSON input {path.name!r} must contain an object.")
    source = path.name
    default_script = path.stem
    runs = content.get("runs")
    observations: list[MemoryObservation] = []
    if isinstance(runs, list):
        for item in runs:
            if isinstance(item, Mapping):
                if "observation_count" in content and item.get("source") != source:
                    continue
                observations.append(_observation_from_mapping(source, cast(Mapping[str, object], item), default_script=default_script))
        return observations

    # Correctness records and a top-level memory object use different schemas;
    # retain only actual numeric fields that are present rather than inventing a
    # record from an unrelated JSON artifact.
    if "pytorch_peak_allocated_mib" in content or "pytorch_peak_reserved_mib" in content:
        allocator = content.get("allocator")
        allocator_mapping = allocator if isinstance(allocator, Mapping) else {}
        observation_mapping: dict[str, object] = {
            "script": default_script,
            "status": content.get("status", "unknown"),
            "peak_allocated_mib": content.get("pytorch_peak_allocated_mib"),
            "peak_reserved_mib": content.get("pytorch_peak_reserved_mib"),
            "allocator_fraction": allocator_mapping.get("allocator_fraction"),
            "allocator_limit_mib": allocator_mapping.get("allocator_limit_mib"),
        }
        observations.append(_observation_from_mapping(source, observation_mapping, default_script=default_script))
    elif isinstance(content.get("records"), list):
        for item in cast(list[object], content["records"]):
            if isinstance(item, Mapping):
                observations.append(_observation_from_mapping(source, cast(Mapping[str, object], item), default_script=default_script))
    return observations


def _read_csv_observations(path: Path) -> list[MemoryObservation]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
    except OSError as error:
        raise ValueError(f"Cannot read CSV input {path.name!r}.") from error
    return [
        _observation_from_mapping(path.name, cast(Mapping[str, object], row), default_script=path.stem)
        for row in rows
    ]


def read_observations(path: Path) -> list[MemoryObservation]:
    """Read public, lightweight artifacts without using filenames as private paths."""

    if path.suffix.lower() == ".json":
        return _read_json_observations(path)
    if path.suffix.lower() == ".csv":
        return _read_csv_observations(path)
    raise ValueError(f"Unsupported input type for {path.name!r}; use CSV or JSON.")


def _default_inputs(input_dir: Path) -> list[Path]:
    return [input_dir / name for name in KNOWN_RESULT_FILES if (input_dir / name).exists()]


def _deduplicate(observations: Iterable[MemoryObservation]) -> list[MemoryObservation]:
    """Remove exact duplicates when both a sidecar and a CSV are supplied."""

    unique: dict[tuple[object, ...], MemoryObservation] = {}
    for observation in observations:
        key = (
            observation.script,
            observation.status,
            observation.formal,
            observation.peak_allocated_mib,
            observation.peak_reserved_mib,
            observation.allocator_fraction,
            observation.allocator_limit_mib,
        )
        unique.setdefault(key, observation)
    return list(unique.values())


def summarize(observations: Sequence[MemoryObservation]) -> dict[str, object]:
    """Create the required evidence contract from actual observed numeric peaks."""

    allocated = [observation.peak_allocated_mib for observation in observations if observation.peak_allocated_mib is not None]
    reserved = [observation.peak_reserved_mib for observation in observations if observation.peak_reserved_mib is not None]
    max_allocated = max(allocated) if allocated else None
    max_reserved = max(reserved) if reserved else None
    allocator_fraction = next(
        (observation.allocator_fraction for observation in reversed(observations) if observation.allocator_fraction is not None),
        None,
    )
    allocator_limit = next(
        (observation.allocator_limit_mib for observation in reversed(observations) if observation.allocator_limit_mib is not None),
        None,
    )
    successful_observations = [observation for observation in observations if observation.status == "success"]
    missing_allocator_evidence = [
        observation
        for observation in successful_observations
        if observation.allocator_fraction is None or observation.allocator_limit_mib is None
    ]
    allocator_evidence_complete = not missing_allocator_evidence
    if max_reserved is None:
        status = "unavailable"
        within_allocator_guard: bool | None = None
    elif not allocator_evidence_complete:
        status = "incomplete_evidence"
        within_allocator_guard = None
    else:
        status = "success"
        within_allocator_guard = max_reserved <= ALLOCATOR_LIMIT_MIB
    return {
        "schema_version": 1,
        "status": status,
        "allocator": {
            "allocator_fraction": allocator_fraction,
            "allocator_limit_mib": allocator_limit,
            "target_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": HARD_MEMORY_LIMIT_MIB,
        "pytorch_peak_allocated_mib": round(max_allocated, 3) if max_allocated is not None else None,
        "pytorch_peak_reserved_mib": round(max_reserved, 3) if max_reserved is not None else None,
        "allocator_evidence_complete": allocator_evidence_complete,
        "missing_allocator_evidence_count": len(missing_allocator_evidence),
        "within_allocator_guard": within_allocator_guard,
        "within_hard_limit": None if max_reserved is None else max_reserved <= HARD_MEMORY_LIMIT_MIB,
        "within_24gib": within_allocator_guard,
        "observation_count": len(observations),
        "runs": [observation.as_json() for observation in observations],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_output_dir(), help="Directory containing local A2-K result artifacts.")
    parser.add_argument("--input", type=Path, action="append", default=[], help="Additional CSV/JSON artifact; repeat as needed.")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: <input-dir>/memory_evidence.json).")
    return parser


def run(*, input_dir: Path, inputs: Sequence[Path], output: Path | None) -> int:
    output_path = output if output is not None else input_dir / "memory_evidence.json"
    selected_inputs = list(inputs) if inputs else _default_inputs(input_dir)
    # When re-summarizing in place, use the prior evidence file as an input only
    # if it already exists.  Its run-level observations are still actual data.
    observations: list[MemoryObservation] = []
    errors: list[str] = []
    for path in selected_inputs:
        if not path.exists():
            errors.append(f"Missing input artifact: {path.name}")
            continue
        try:
            observations.extend(read_observations(path))
        except ValueError as error:
            errors.append(str(error))
    all_observations = _deduplicate(observations)
    observations = [observation for observation in all_observations if observation.formal is True]
    payload = summarize(observations)
    payload["excluded_nonformal_observation_count"] = len(all_observations) - len(observations)
    if errors:
        payload["input_errors"] = errors
    if not observations:
        payload["reason"] = "No formal artifact contained an observed PyTorch peak; no memory value was fabricated."
    write_json(output_path, payload)
    if not observations:
        stderr("No formal observed memory peaks were found; memory_evidence.json was written as unavailable.")
        return 2
    if payload["allocator_evidence_complete"] is not True:
        stderr(
            "Formal success observations are missing allocator_fraction or allocator_limit_mib; "
            "memory_evidence.json was written as incomplete evidence."
        )
        return 1
    if errors:
        stderr("Some input artifacts could not be read; inspect input_errors in memory_evidence.json.")
        return 1
    print(f"Wrote memory evidence from {len(observations)} observations to {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(input_dir=args.input_dir, inputs=args.input, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
