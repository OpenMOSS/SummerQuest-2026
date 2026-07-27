from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


RESULT_FILES = ("correctness.json", "unit_tests.txt", "checkpointing.csv", "attention_baseline.csv", "compile_comparison.csv", "flash_benchmark.csv", "run_metadata.json")


def sanitize_text(text: str) -> str:
    text = re.sub(r"/[^\s,\"]*/assignment1-basics/\.venv/bin/python", "python", text)
    text = re.sub(
        r"/[^\s,\"]*/assignment2-systems/student_scripts/a2k/([A-Za-z0-9_]+)\.py",
        r"python -m student_scripts.a2k.\1",
        text,
    )
    text = re.sub(r"/[^\s,\"]*/assignment2-systems", "WORKSPACE", text)
    return text


def sanitize_file(path: Path) -> None:
    path.write_text(sanitize_text(path.read_text(encoding="utf-8")), encoding="utf-8")


def enrich_speedups(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baselines = {
        (row["sequence_length"], row["head_dim"], row["dtype"], row["causal"], row["phase"]): float(row["p50_ms"])
        for row in rows if row["implementation"] == "eager" and row["status"] == "success"
    }
    for row in rows:
        key = (row["sequence_length"], row["head_dim"], row["dtype"], row["causal"], row["phase"])
        if row["status"] == "success" and key in baselines:
            row["speedup_vs_eager"] = baselines[key] / float(row["p50_ms"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def memory_evidence(output: Path) -> None:
    peaks = []
    for path in output.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("peak_reserved_mib"):
                    identity = {key: row[key] for key in ("config_id", "implementation", "sequence_length", "head_dim", "phase") if row.get(key)}
                    peaks.append({"source": path.name, **identity, "peak_allocated_mib": float(row["peak_allocated_mib"]), "peak_reserved_mib": float(row["peak_reserved_mib"])})
    maximum_allocated = max((row["peak_allocated_mib"] for row in peaks), default=0)
    maximum_reserved = max((row["peak_reserved_mib"] for row in peaks), default=0)
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    value = {
        "allocator": {
            "allocator_fraction": float(metadata["allocator_fraction"]),
            "allocator_limit_mib": 23552,
        },
        "hard_limit_mib": 24576,
        "pytorch_peak_allocated_mib": maximum_allocated,
        "pytorch_peak_reserved_mib": maximum_reserved,
        "within_24gib": maximum_reserved <= 23552,
        "runs": peaks,
    }
    (output / "memory_evidence.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--output", type=Path, default=Path("results/a2k_submission"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in RESULT_FILES:
        source = args.raw_root / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output / name)
        sanitize_file(args.output / name)
    enrich_speedups(args.output / "flash_benchmark.csv")
    memory_evidence(args.output)


if __name__ == "__main__":
    main()
