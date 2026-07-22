"""Create a compact, path-sanitized summary from training JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="One or more metrics.jsonl files")
    parser.add_argument("--output", help="Write the summary instead of printing it")
    parser.add_argument("--format", choices=("json", "markdown"), default=None)
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: each line must contain a JSON object")
            records.append(record)
    return records


def summarize(path: Path) -> dict[str, Any]:
    records = _read_records(path)
    training = [record for record in records if record.get("event") == "train"]
    validation = [record for record in records if record.get("event") == "validation"]
    endings = [record for record in records if record.get("event") == "run_end"]
    starts = [record for record in records if record.get("event") == "run_start"]

    def finite_losses(items: list[dict[str, Any]]) -> list[float]:
        losses: list[float] = []
        for item in items:
            value = item.get("loss")
            if isinstance(value, (int, float)):
                losses.append(float(value))
        return losses

    train_losses = finite_losses(training)
    validation_losses = finite_losses(validation)
    final = endings[-1] if endings else (training[-1] if training else {})
    start = starts[-1] if starts else {}
    return {
        "run": path.parent.name or path.stem,
        "config_hash": start.get("config_hash", final.get("config_hash")),
        "device": start.get("device"),
        "seed": start.get("seed"),
        "num_parameters": start.get("num_parameters"),
        "final_iteration": final.get("iteration"),
        "processed_tokens": final.get("processed_tokens"),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "best_train_loss": min(train_losses) if train_losses else None,
        "final_validation_loss": validation_losses[-1] if validation_losses else None,
        "best_validation_loss": min(validation_losses) if validation_losses else None,
        "wall_time_seconds": final.get("wall_time_seconds"),
        "status": "complete" if endings else "incomplete",
    }


def _markdown(summaries: list[dict[str, Any]]) -> str:
    headers = (
        "run",
        "status",
        "final_iteration",
        "processed_tokens",
        "final_train_loss",
        "best_validation_loss",
        "wall_time_seconds",
    )

    def display(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return "-" if value is None else str(value).replace("|", "\\|")

    lines: list[str] = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(display(summary.get(header)) for header in headers) + " |" for summary in summaries)
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summaries = [summarize(Path(value).expanduser()) for value in args.logs]
    output_format = args.format or ("markdown" if args.output and Path(args.output).suffix.lower() in {".md", ".markdown"} else "json")
    rendered = _markdown(summaries) if output_format == "markdown" else json.dumps(summaries, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
