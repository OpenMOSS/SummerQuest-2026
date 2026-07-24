#!/usr/bin/env python3
"""Collect sanitized A1 experiment results into a submission-ready logs tree.

Only small, human-readable result files are copied. Checkpoints, raw scheduler
logs, machine details, and internal job/resource identifiers are intentionally
excluded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024

METRIC_KEYS = (
    "event",
    "reason",
    "status",
    "step",
    "wall_clock_sec",
    "train_loss",
    "val_loss",
    "lr",
    "processed_tokens",
)
BATCH_KEYS = (
    "status",
    "batch_size",
    "step",
    "wall_clock_sec",
    "train_loss",
    "lr",
    "processed_tokens",
    "tokens_per_sec",
    "peak_memory_bytes",
)
RUN_SUMMARY_KEYS = (
    "run_name",
    "status",
    "divergence_reason",
    "completed_steps",
    "processed_tokens",
    "total_training_time_sec",
    "final_train_loss",
    "final_val_loss",
    "best_val_loss",
    "parameter_count",
    "model",
    "training",
)

_PRIVATE_KEYS = {
    "accelerator",
    "accelerators",
    "compute_group",
    "compute_group_id",
    "container_id",
    "cpu",
    "cpu_count",
    "cpu_model",
    "cuda_visible_devices",
    "device",
    "device_id",
    "device_name",
    "device_type",
    "devices",
    "gpu",
    "gpu_id",
    "gpus",
    "hardware",
    "host",
    "hostname",
    "image",
    "image_id",
    "instance",
    "instances",
    "internal_id",
    "job",
    "job_id",
    "job_url",
    "local_rank",
    "machine",
    "machine_id",
    "node",
    "node_id",
    "pod",
    "pod_id",
    "platform",
    "priority",
    "project",
    "project_id",
    "qz_job_id",
    "rank",
    "replicas",
    "resource",
    "resource_id",
    "resources",
    "runtime_environment",
    "environment",
    "scheduler",
    "spec",
    "spec_id",
    "task_id",
    "workspace",
    "workspace_id",
    "world_size",
}
_PRIVATE_KEY_PREFIXES = (
    "accelerator_",
    "compute_group_",
    "container_",
    "cpu_",
    "cuda_",
    "device_",
    "gpu_",
    "hardware_",
    "host_",
    "instance_",
    "image_",
    "job_",
    "node_",
    "machine_",
    "pod_",
    "platform_",
    "project_",
    "qz_",
    "resource_",
    "scheduler_",
    "spec_",
    "task_",
    "workspace_",
)
_PRIVATE_COMPACT_KEYS = {
    "computegroupid",
    "containerid",
    "deviceid",
    "gpuid",
    "hostid",
    "instanceid",
    "jobid",
    "jobuuid",
    "nodeid",
    "podid",
    "projectid",
    "qzjobid",
    "resourceid",
    "specid",
    "taskid",
    "workspaceid",
}
_FREE_TEXT_KEYS = {"comment", "generated_text", "prompt", "sample", "text"}
_SKIP_PARTS = {"checkpoint", "checkpoints", "orchestration", "qzcli", "scheduler"}
_TOKENIZER_HINTS = ("metadata", "metric", "summary", "tokenizer")
_GENERATION_HINTS = ("generation", "generated", "sample")
_UUID_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?![0-9a-f])")
_LONG_HEX_ID_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{20,}(?![0-9a-f])")
_FILE_URI_RE = re.compile(r"(?i)\bfile:///[^\s,;:)\]}]+")
_HOME_PATH_RE = re.compile(r"(?<![\w])~/(?:[^/\s]+/)*[^\s,;:)\]}]*")
_UNIX_PATH_RE = re.compile(r"(?<![\w:/])/(?:[^/\s]+/)+[^\s,;:)\]}]*")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:[^\\\r\n]+\\)*[^\\\r\n\s,;:)\]}]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Experiment runs directory.")
    parser.add_argument(
        "--tokenizer-artifacts",
        action="append",
        type=Path,
        default=None,
        metavar="DIR",
        help="Tokenizer result directory; may be repeated (default: tokenizer_artifacts).",
    )
    parser.add_argument(
        "--generations",
        action="append",
        type=Path,
        default=None,
        metavar="DIR",
        help="Generated-sample directory; may be repeated (default: generations).",
    )
    parser.add_argument(
        "--artifacts",
        action="append",
        type=Path,
        default=[],
        metavar="DIR",
        help="Additional mixed tokenizer/generation artifact directory; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination logs directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after a complete result tree has been staged.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def is_private_key(key: object) -> bool:
    normalized = normalized_key(key)
    private_suffixes = (
        "_job_id",
        "_job_uuid",
        "_node_id",
        "_pod_id",
        "_project_id",
        "_resource_id",
        "_task_id",
        "_workspace_id",
    )
    compact = normalized.replace("_", "")
    return (
        normalized in _PRIVATE_KEYS
        or compact in _PRIVATE_COMPACT_KEYS
        or normalized.startswith(_PRIVATE_KEY_PREFIXES)
        or normalized.endswith(private_suffixes)
    )


def sanitize_string(value: str, key: str | None) -> str:
    if key == "bytes_hex" and re.fullmatch(r"[0-9a-fA-F]*", value):
        # Token byte strings are experimental evidence, not opaque identifiers.
        return value
    is_absolute = Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    if is_absolute:
        if key in _FREE_TEXT_KEYS:
            return "<redacted-path>"
        name = PureWindowsPath(value).name if PureWindowsPath(value).is_absolute() else Path(value).name
        return name or "."
    sanitized = _FILE_URI_RE.sub("<redacted-path>", value)
    sanitized = _HOME_PATH_RE.sub("<redacted-path>", sanitized)
    sanitized = _UNIX_PATH_RE.sub("<redacted-path>", sanitized)
    sanitized = _WINDOWS_PATH_RE.sub("<redacted-path>", sanitized)
    sanitized = _UUID_RE.sub("<redacted-id>", sanitized)
    return _LONG_HEX_ID_RE.sub("<redacted-id>", sanitized)


def sanitize(value: Any, key: str | None = None) -> Any:
    """Recursively remove non-submission metadata and absolute path prefixes."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, child in value.items():
            if is_private_key(raw_key):
                continue
            child_key = normalized_key(raw_key)
            result[str(raw_key)] = sanitize(child, child_key)
        return result
    if isinstance(value, list):
        return [sanitize(item, key) for item in value]
    if isinstance(value, str):
        return sanitize_string(value, key)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def project_fields(record: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    return sanitize({key: record[key] for key in allowed if key in record})


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON result {path}: {error}") from error


def read_jsonl(path: Path, allowed: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read JSONL result {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"expected an object in {path} at line {line_number}")
        records.append(project_fields(record, allowed) if allowed is not None else sanitize(record))
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def slug(value: str) -> str:
    """Return a filesystem-safe public label with identifier-shaped text removed."""

    value = _UUID_RE.sub("redacted-id", value)
    value = _LONG_HEX_ID_RE.sub("redacted-id", value)
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "result"


def skip_source(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool(parts & _SKIP_PARTS) or any("checkpoint" in part for part in parts)


def result_name(metrics_path: Path, runs: Path, summary: Any) -> str:
    if isinstance(summary, dict) and isinstance(summary.get("run_name"), str):
        return slug(summary["run_name"])
    relative_parent = metrics_path.parent.relative_to(runs)
    return slug("__".join(relative_parent.parts))


def training_destination(name: str, metrics_path: Path, runs: Path) -> Path:
    lower_name = name.lower()
    relative_parts = {part.lower() for part in metrics_path.relative_to(runs).parts}
    if lower_name == "tinystories_baseline":
        return Path("train_tinystories.jsonl")
    if lower_name == "owt_baseline":
        return Path("train_owt.jsonl")
    if lower_name.startswith("ablation_"):
        return Path(f"{name}.jsonl")
    if "lr_sweep" in relative_parts or lower_name.startswith("lr_"):
        return Path("lr_sweep") / f"{name}.jsonl"
    return Path("training") / f"{name}.jsonl"


def batch_destination(path: Path, runs: Path) -> Path:
    relative = path.relative_to(runs)
    if relative.parent == Path("batch_size") and relative.stem == "summary":
        return Path("batch_size") / relative.name
    flattened = slug("__".join(relative.with_suffix("").parts))
    return Path("batch_size") / f"{flattened}{path.suffix.lower()}"


def compact_run_summary(summary: Any) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return None
    return project_fields(summary, RUN_SUMMARY_KEYS)


def reserve(destination: Path, source: Path, reserved: dict[Path, Path]) -> None:
    previous = reserved.get(destination)
    if previous is not None and previous != source:
        raise ValueError(f"multiple inputs map to {destination}: {previous} and {source}")
    reserved[destination] = source


def is_generation(path: Path, value: Any) -> bool:
    lowered = path.name.lower()
    if any(hint in lowered for hint in _GENERATION_HINTS):
        return True
    return (
        isinstance(value, dict)
        and "text" in value
        and ("generated_new_tokens" in value or "requested_new_tokens" in value)
    )


def is_tokenizer_result(path: Path, value: Any) -> bool:
    if path.name.lower() == "vocab.json":
        return False
    lowered = path.name.lower()
    if any(hint in lowered for hint in _TOKENIZER_HINTS):
        return True
    if not isinstance(value, dict):
        return False
    return "tokenizer" in value or value.get("artifact_format") == "gpt2_byte_level_bpe"


def artifact_destination(category: str, path: Path, root: Path) -> Path:
    relative = path.relative_to(root)
    safe_parts = [slug(part) for part in relative.parts]
    return Path(category, *safe_parts)


def validate_public_tree(output: Path) -> None:
    banned_suffixes = {".bin", ".ckpt", ".npz", ".npy", ".pt", ".pth", ".safetensors"}
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"refusing to publish a symbolic link: {path.relative_to(output)}")
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        lowered_parts = {part.lower() for part in relative.parts}
        if path.suffix.lower() in banned_suffixes or any("checkpoint" in part for part in lowered_parts):
            raise ValueError(f"refusing to publish a checkpoint or binary artifact: {relative}")
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_BYTES:
            raise ValueError(f"public result exceeds the 5 MiB per-file limit: {relative} ({size} bytes)")


def main() -> int:
    args = parse_args()
    runs = resolve(args.runs.expanduser()).resolve()
    tokenizer_roots = [
        resolve(path.expanduser()).resolve() for path in (args.tokenizer_artifacts or [Path("tokenizer_artifacts")])
    ]
    generation_roots = [resolve(path.expanduser()).resolve() for path in (args.generations or [Path("generations")])]
    mixed_roots = [resolve(path.expanduser()).resolve() for path in args.artifacts]
    output = resolve(args.output_dir.expanduser()).resolve()

    source_specs: list[tuple[str, Path]] = []
    seen_roots: set[Path] = set()

    def add_source(kind: str, path: Path) -> None:
        if path not in seen_roots:
            source_specs.append((kind, path))
            seen_roots.add(path)

    add_source("runs", runs)
    for source_root in tokenizer_roots:
        add_source("tokenizer", source_root)
    for source_root in generation_roots:
        add_source("generation", source_root)
    for source_root in mixed_roots:
        add_source("mixed", source_root)

    existing_roots = [path for _, path in source_specs if path.is_dir()]
    if not existing_roots:
        raise FileNotFoundError("none of the run, tokenizer, generation, or mixed artifact directories exists")
    for source_root in existing_roots:
        if is_relative_to(output, source_root) or is_relative_to(source_root, output):
            raise ValueError("--output-dir must not contain, or be contained by, any input directory")
    if output.is_symlink():
        raise ValueError("--output-dir must not be a symbolic link")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it atomically")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        reserved: dict[Path, Path] = {}
        training_runs: list[dict[str, Any]] = []
        consumed_json: set[Path] = set()

        if runs.is_dir():
            for metrics_path in sorted(runs.rglob("metrics.jsonl")):
                if skip_source(metrics_path):
                    continue
                summary_path = metrics_path.with_name("summary.json")
                summary = read_json(summary_path) if summary_path.is_file() else None
                if summary_path.is_file():
                    consumed_json.add(summary_path.resolve())
                name = result_name(metrics_path, runs, summary)
                relative_destination = training_destination(name, metrics_path, runs)
                destination = staging / relative_destination
                reserve(destination, metrics_path, reserved)
                write_jsonl(destination, read_jsonl(metrics_path, METRIC_KEYS))

                entry: dict[str, Any] = {
                    "name": name,
                    "metrics_file": relative_destination.as_posix(),
                }
                compact_summary = compact_run_summary(summary)
                if compact_summary is not None:
                    summary_destination = relative_destination.with_suffix(".summary.json")
                    reserve(staging / summary_destination, summary_path, reserved)
                    write_json(staging / summary_destination, compact_summary)
                    entry["summary_file"] = summary_destination.as_posix()
                    entry.update(compact_summary)
                training_runs.append(entry)

        batch_files: list[str] = []
        if runs.is_dir():
            for batch_path in sorted(runs.rglob("*.jsonl")):
                if batch_path.name == "metrics.jsonl" or skip_source(batch_path):
                    continue
                relative = batch_path.relative_to(runs)
                if "batch" not in relative.as_posix().lower():
                    continue
                relative_destination = batch_destination(batch_path, runs)
                reserve(staging / relative_destination, batch_path, reserved)
                write_jsonl(staging / relative_destination, read_jsonl(batch_path, BATCH_KEYS))
                batch_files.append(relative_destination.as_posix())

        validation_file: str | None = None
        validation_path = runs / "orchestration" / "validation.json"
        if validation_path.is_file():
            validation_destination = Path("validation.json")
            write_json(staging / validation_destination, sanitize(read_json(validation_path)))
            validation_file = validation_destination.as_posix()

        tokenizer_files: list[str] = []
        generation_files: list[str] = []
        other_summary_files: list[str] = []
        for kind, source_root in source_specs:
            if not source_root.is_dir():
                continue
            for source in sorted(source_root.rglob("*.json")):
                if source.resolve() in consumed_json or skip_source(source):
                    continue
                value = read_json(source)
                relative_lower = source.relative_to(source_root).as_posix().lower()
                relative_destination: Path | None = None
                destination_group: list[str] | None = None

                if kind == "runs" and "batch" in relative_lower and source.name.startswith("summary"):
                    relative_destination = batch_destination(source, runs)
                    destination_group = batch_files
                elif kind == "generation" or is_generation(source, value):
                    relative_destination = artifact_destination("generation", source, source_root)
                    destination_group = generation_files
                elif kind == "tokenizer" and is_tokenizer_result(source, value):
                    relative_destination = artifact_destination("tokenizer", source, source_root)
                    destination_group = tokenizer_files
                elif kind in {"runs", "mixed"} and is_tokenizer_result(source, value):
                    relative_destination = artifact_destination("tokenizer", source, source_root)
                    destination_group = tokenizer_files
                elif kind == "runs" and source.name == "summary.json":
                    relative_destination = artifact_destination("summaries", source, source_root)
                    destination_group = other_summary_files

                if relative_destination is None or destination_group is None:
                    continue
                reserve(staging / relative_destination, source, reserved)
                write_json(staging / relative_destination, sanitize(value))
                destination_group.append(relative_destination.as_posix())

        training_runs.sort(key=lambda item: str(item["name"]))
        result = {
            "schema_version": 1,
            "training_runs": training_runs,
            "batch_size_files": sorted(set(batch_files)),
            "tokenizer_files": sorted(set(tokenizer_files)),
            "generation_files": sorted(set(generation_files)),
            "other_summary_files": sorted(set(other_summary_files)),
            "validation_file": validation_file,
        }
        write_json(staging / "summary.json", result)
        validate_public_tree(staging)

        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        staging.replace(output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    raise SystemExit(main())
