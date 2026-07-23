from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from profiling.common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize profiling JSON outputs into CSV and Markdown tables.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/summary")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> list[str]:
    if df.empty:
        return []
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    return [str(csv_path), str(md_path)]


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    benchmark_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    accumulation_rows: list[dict[str, Any]] = []
    toy_rows: list[dict[str, Any]] = []

    for path in sorted(results_root.rglob("*.json")):
        payload = load_json(path)
        artifact_type = payload.get("artifact_type")

        if artifact_type == "benchmark":
            run_config = payload["run_config"]
            summary = payload["summaries"]["total_ms"]
            benchmark_rows.append(
                {
                    "source_file": str(path),
                    "model_size": run_config["model_size"],
                    "batch_size": run_config["batch_size"],
                    "context_length": run_config["context_length"],
                    "mode": run_config["mode"],
                    "dtype": run_config["dtype"],
                    "warmup": run_config["warmup"],
                    "steps": run_config["steps"],
                    "mean_ms": summary["mean_ms"],
                    "std_ms": summary["std_ms"],
                    "cv": summary["cv"],
                    "forward_mean_ms": payload["summaries"]["forward_ms"]["mean_ms"],
                    "backward_mean_ms": payload["summaries"]["backward_ms"]["mean_ms"],
                    "optimizer_mean_ms": payload["summaries"]["optimizer_ms"]["mean_ms"],
                }
            )
        elif artifact_type == "memory_snapshot":
            run_config = payload["run_config"]
            memory = payload["memory"]
            memory_rows.append(
                {
                    "source_file": str(path),
                    "snapshot_path": payload["snapshot_path"],
                    "model_size": run_config["model_size"],
                    "context_length": run_config["context_length"],
                    "mode": run_config["mode"],
                    "dtype": run_config["dtype"],
                    "active_peak_bytes": memory["active_peak_bytes"],
                    "allocated_peak_bytes": memory["allocated_peak_bytes"],
                    "reserved_peak_bytes": memory["reserved_peak_bytes"],
                    "requested_peak_bytes": memory["requested_peak_bytes"],
                }
            )
        elif artifact_type == "mixed_precision_accumulation":
            for case in payload["cases"]:
                accumulation_rows.append(case)
        elif artifact_type == "toy_autocast_dtype_inspection":
            toy_rows.append(payload)

    generated_files: list[str] = []
    generated_files.extend(save_table(pd.DataFrame(benchmark_rows), output_dir, "benchmark_summary"))
    generated_files.extend(save_table(pd.DataFrame(memory_rows), output_dir, "memory_summary"))
    generated_files.extend(save_table(pd.DataFrame(accumulation_rows), output_dir, "mixed_precision_accumulation"))
    generated_files.extend(save_table(pd.DataFrame(toy_rows), output_dir, "toy_autocast_dtypes"))

    index_payload = {
        "artifact_type": "summary_index",
        "results_root": str(results_root),
        "generated_files": generated_files,
        "counts": {
            "benchmark": len(benchmark_rows),
            "memory_snapshot": len(memory_rows),
            "mixed_precision_accumulation": len(accumulation_rows),
            "toy_autocast_dtype_inspection": len(toy_rows),
        },
    }
    index_path = output_dir / "summary_index.json"
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved summaries under {output_dir}")


if __name__ == "__main__":
    main()
