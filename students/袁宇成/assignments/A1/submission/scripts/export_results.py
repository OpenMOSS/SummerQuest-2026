#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def copy_run(source: Path, destination: Path) -> dict | None:
    metrics = source / "metrics.jsonl"
    summary = source / "summary.json"
    if not metrics.is_file() or not summary.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics, destination.with_suffix(".jsonl"))
    shutil.copy2(summary, destination.with_name(destination.name + "_summary.json"))
    return json.loads(summary.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact logs and plots for the public A1 report.")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    logs = args.destination / "logs"
    assets = args.destination / "assets"
    logs.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    main_sources = {
        "tinystories": args.artifacts / "runs/tinystories_baseline",
        "owt": args.artifacts / "runs/owt_baseline",
    }
    for name, source in main_sources.items():
        summary = copy_run(source, logs / f"train_{name}")
        if summary is not None:
            summaries[name] = summary

    groups = {"lr_sweep": "lr_*", "batch_size": "batch_*", "ablations": "ablation_*"}
    for group, pattern in groups.items():
        for source in sorted((args.artifacts / "experiments").glob(pattern)):
            summary = copy_run(source, logs / group / source.name)
            if summary is not None:
                summaries[source.name] = summary

    tokenizer_results = {}
    for path in sorted((args.artifacts / "tokenizer_analysis").glob("*.json")):
        destination = logs / "tokenizers" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        tokenizer_results[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    summaries["tokenizers"] = tokenizer_results
    (logs / "summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("plot_results.py")), "--artifacts", str(args.artifacts), "--output-dir", str(assets)],
        check=True,
    )


if __name__ == "__main__":
    main()
