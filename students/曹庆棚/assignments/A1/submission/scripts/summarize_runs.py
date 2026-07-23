from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "run_name",
    "status",
    "steps",
    "batch_size",
    "context_length",
    "processed_tokens",
    "wall_clock_sec",
    "tokens_per_sec",
    "peak_gpu_memory_bytes",
    "final_train_loss",
    "final_val_loss",
    "best_val_loss",
    "best_step",
    "best_checkpoint",
    "max_lr",
    "min_lr",
    "warmup_iters",
    "cosine_cycle_iters",
    "ablation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect per-run summary.json files into comparison tables.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, help="Optional GitHub-flavored Markdown comparison table.")
    return parser.parse_args()


def _resolve_summary_path(path: Path) -> Path:
    return path / "summary.json" if path.is_dir() else path


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def _last_value(records: list[dict[str, Any]], key: str) -> Any:
    return next((record[key] for record in reversed(records) if key in record), None)


def _best_validation(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if record.get("val_loss") is not None]
    return min(candidates, key=lambda record: float(record["val_loss"]), default=None)


def _product(*values: Any) -> int | None:
    if any(value is None for value in values):
        return None
    return int(values[0]) * int(values[1]) * int(values[2])


def normalize_summary(path: Path) -> dict[str, Any]:
    summary_path = _resolve_summary_path(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"{summary_path}: summary must be a JSON object")

    records = _read_records(summary_path.with_name("train.jsonl"))
    best_record = _best_validation(records)
    optimizer = summary.get("optimizer") or {}
    if not isinstance(optimizer, dict):
        optimizer = {}
    ablation = summary.get("ablation") or {}

    steps = summary.get("steps", _last_value(records, "step"))
    batch_size = summary.get("batch_size")
    context_length = summary.get("context_length")
    processed_tokens = summary.get("processed_tokens")
    if processed_tokens is None:
        processed_tokens = _product(steps, batch_size, context_length)

    best_val_loss = summary.get("best_val_loss")
    best_step = summary.get("best_step")
    if best_record is not None:
        best_val_loss = best_val_loss if best_val_loss is not None else best_record["val_loss"]
        best_step = best_step if best_step is not None else best_record.get("step")

    wall_clock_sec = summary.get("wall_clock_sec")
    tokens_per_sec = summary.get("tokens_per_sec")
    if tokens_per_sec is None and processed_tokens is not None and wall_clock_sec:
        tokens_per_sec = float(processed_tokens) / float(wall_clock_sec)

    return {
        "run_name": summary.get("run_name", summary_path.parent.name),
        "status": summary.get("status", "unknown"),
        "steps": steps,
        "batch_size": batch_size,
        "context_length": context_length,
        "processed_tokens": processed_tokens,
        "wall_clock_sec": wall_clock_sec,
        "tokens_per_sec": tokens_per_sec,
        "peak_gpu_memory_bytes": summary.get("peak_gpu_memory_bytes"),
        "final_train_loss": summary.get("final_train_loss", _last_value(records, "train_loss")),
        "final_val_loss": summary.get("final_val_loss", _last_value(records, "val_loss")),
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "best_checkpoint": summary.get("best_checkpoint"),
        "max_lr": optimizer.get("max_lr", summary.get("max_lr")),
        "min_lr": optimizer.get("min_lr", summary.get("min_lr")),
        "warmup_iters": optimizer.get("warmup_iters", summary.get("warmup_iters")),
        "cosine_cycle_iters": optimizer.get("cosine_cycle_iters", summary.get("cosine_cycle_iters")),
        "ablation": json.dumps(ablation, ensure_ascii=False, sort_keys=True),
    }


def _markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(records: list[dict[str, Any]], path: Path) -> None:
    preferred_fields = (
        "run_name",
        "status",
        "steps",
        "batch_size",
        "processed_tokens",
        "max_lr",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "best_step",
        "tokens_per_sec",
        "peak_gpu_memory_bytes",
        "ablation",
    )
    lines = [
        "| " + " | ".join(preferred_fields) + " |",
        "| " + " | ".join("---" for _ in preferred_fields) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_value(record[field]) for field in preferred_fields) + " |" for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = [normalize_summary(run) for run in args.runs]

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    if args.markdown is not None:
        write_markdown(records, args.markdown)


if __name__ == "__main__":
    main()
