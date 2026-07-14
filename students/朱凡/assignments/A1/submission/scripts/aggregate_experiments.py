"""Aggregate manifest entries, metrics, and Slurm state into submission logs."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "experiment_manifest.jsonl"
RUNS_DIR = REPO_ROOT / "data" / "runs"
OUTPUT_DIR = REPO_ROOT / "data" / "submission_logs"


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                records.append({"parse_error": line.rstrip("\n")})
    return records


def _slurm_state(job_id: str) -> dict[str, str | None]:
    try:
        output = subprocess.check_output(
            ["sacct", "-n", "-P", "-j", job_id, "--format=State,ExitCode,Elapsed,MaxRSS"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"state": None, "exit_code": None, "slurm_elapsed": None, "max_rss": None}
    first = next((line for line in output.splitlines() if line.strip()), "")
    fields = (first.split("|") + [None] * 4)[:4]
    return dict(zip(["state", "exit_code", "slurm_elapsed", "max_rss"], fields, strict=True))


def _summary(run_name: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    metrics = _read_json_lines(run_dir / "metrics.jsonl")
    validation = [record for record in metrics if isinstance(record.get("validation_loss"), (int, float))]
    train = [record for record in metrics if isinstance(record.get("train_loss"), (int, float))]
    numeric_values = [value for record in metrics for value in record.values() if isinstance(value, float)]
    submitted_jobs = [event for event in events if event.get("job_id")]
    states = [{**event, **_slurm_state(str(event["job_id"]))} for event in submitted_jobs]
    last_validation = validation[-1] if validation else {}
    best_validation = min(validation, key=lambda record: record["validation_loss"]) if validation else {}
    return {
        "run_name": run_name,
        "attempts": states,
        "has_metrics": bool(metrics),
        "metric_parse_errors": sum("parse_error" in record for record in metrics),
        "has_non_finite_metric": any(not math.isfinite(value) for value in numeric_values),
        "final_iteration": last_validation.get("iteration"),
        "final_validation_loss": last_validation.get("validation_loss"),
        "best_validation_loss": best_validation.get("validation_loss"),
        "best_validation_iteration": best_validation.get("iteration"),
        "best_validation_tokens": best_validation.get("tokens_seen"),
        "best_validation_process_seconds": best_validation.get("process_elapsed_seconds"),
        "final_train_loss": train[-1].get("train_loss") if train else None,
        "processed_tokens": last_validation.get("tokens_seen"),
        "training_elapsed_seconds": last_validation.get("training_elapsed_seconds"),
        "process_elapsed_seconds": last_validation.get("process_elapsed_seconds"),
    }


def main() -> None:
    events_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in _read_json_lines(MANIFEST):
        if event.get("run_name"):
            events_by_run[str(event["run_name"])].append(event)
    for run_dir in RUNS_DIR.glob("*") if RUNS_DIR.exists() else []:
        if run_dir.is_dir():
            events_by_run.setdefault(run_dir.name, [])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for run_name, events in sorted(events_by_run.items()):
        source = RUNS_DIR / run_name
        destination = OUTPUT_DIR / run_name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("config.json", "metrics.jsonl"):
            if (source / filename).exists():
                shutil.copy2(source / filename, destination / filename)
        (destination / "summary.json").write_text(
            json.dumps(_summary(run_name, events), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(run_name)


if __name__ == "__main__":
    main()
