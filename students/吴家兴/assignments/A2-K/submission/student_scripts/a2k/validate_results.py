"""Strict local validation for the completed A2-K public artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_timing_rows(
    rows: list[dict[str, str]],
    *,
    expected_count: int,
    name: str,
) -> None:
    require(len(rows) == expected_count, f"{name}: wrong row count")
    for row in rows:
        require(row["status"] in {"ok", "oom"}, f"{name}: invalid status")
        require(
            int(float(row["allocator_limit_mib"])) == 23 * 1024,
            f"{name}: allocator limit mismatch",
        )
        require(
            float(row["free_memory_mib_at_start"]) >= 22 * 1024,
            f"{name}: formal free-memory gate failed",
        )
        reserved = row.get("peak_reserved_mib", "")
        if reserved:
            require(
                float(reserved) <= 23 * 1024,
                f"{name}: peak reserved exceeds allocator limit",
            )
        if row["status"] == "ok" and row.get("p50_ms"):
            p20 = float(row["p20_ms"])
            p50 = float(row["p50_ms"])
            p80 = float(row["p80_ms"])
            require(p20 <= p50 <= p80, f"{name}: invalid quantile order")


def main() -> int:
    args = parse_args()
    assignment = args.assignment_dir
    results = assignment / "results"
    assets = assignment / "assets"
    required_files = (
        "correctness.json",
        "unit_tests.txt",
        "checkpointing.csv",
        "attention_baseline.csv",
        "compile_comparison.csv",
        "flash_benchmark.csv",
        "memory_evidence.json",
        "run_metadata.json",
    )
    for relative in required_files:
        require((results / relative).is_file(), f"missing {relative}")

    checkpoint_rows = read_csv(results / "checkpointing.csv")
    require(len(checkpoint_rows) == 7, "checkpointing: wrong row count")
    expected_checkpoint = {
        (1024, block) for block in ("none", "1", "2", "4", "8")
    } | {(2048, "none"), (2048, "2")}
    observed_checkpoint = {
        (int(row["context_length"]), row["checkpoint_block_size"])
        for row in checkpoint_rows
    }
    require(
        observed_checkpoint == expected_checkpoint,
        "checkpointing: configuration matrix mismatch",
    )
    for row in checkpoint_rows:
        require(row["status"] in {"ok", "oom"}, "checkpointing: bad status")
        require(
            int(float(row["allocator_limit_mib"])) == 23 * 1024,
            "checkpointing: allocator mismatch",
        )
        require(
            float(row["free_memory_mib_at_start"]) >= 22 * 1024,
            "checkpointing: free-memory gate failed",
        )
        if row["peak_reserved_mib"]:
            require(
                float(row["peak_reserved_mib"]) <= 23 * 1024,
                "checkpointing: reserved memory too high",
            )

    attention_rows = read_csv(results / "attention_baseline.csv")
    validate_timing_rows(
        attention_rows,
        expected_count=18,
        name="attention_baseline",
    )
    expected_attention = {
        (str(sequence), str(head_dim), phase)
        for sequence in (512, 2048, 8192)
        for head_dim in (64, 128)
        for phase in ("forward", "backward", "forward-backward")
    }
    require(
        {
            (row["seq_len"], row["head_dim"], row["phase"])
            for row in attention_rows
        }
        == expected_attention,
        "attention_baseline: matrix mismatch",
    )

    compile_rows = read_csv(results / "compile_comparison.csv")
    validate_timing_rows(
        compile_rows,
        expected_count=24,
        name="compile_comparison",
    )
    require(
        sum(row["target"] == "attention" for row in compile_rows) == 18,
        "compile_comparison: attention row count",
    )
    require(
        sum(row["target"] == "small-model" for row in compile_rows) == 6,
        "compile_comparison: model row count",
    )

    flash_rows = read_csv(results / "flash_benchmark.csv")
    validate_timing_rows(
        flash_rows,
        expected_count=66,
        name="flash_benchmark",
    )
    require(
        sum(row["implementation"] == "eager" for row in flash_rows) == 24,
        "flash_benchmark: eager row count",
    )
    require(
        sum(row["implementation"] == "compiled" for row in flash_rows) == 18,
        "flash_benchmark: compiled row count",
    )
    require(
        sum(row["implementation"] == "triton" for row in flash_rows) == 24,
        "flash_benchmark: Triton row count",
    )
    for row in flash_rows:
        if row["status"] == "ok":
            require(
                bool(row["speedup_vs_eager"]),
                "flash_benchmark: missing valid speedup",
            )

    correctness = json.loads(
        (results / "correctness.json").read_text(encoding="utf-8")
    )
    require(
        correctness["summary"]
        == {"total": 38, "passed": 38, "failed": 0, "skipped": 0},
        "correctness summary mismatch",
    )
    unit_tests = (results / "unit_tests.txt").read_text(encoding="utf-8")
    require("6 passed" in unit_tests, "official test pass count missing")
    require("FAILED" not in unit_tests, "official test failure present")
    require("SKIPPED" not in unit_tests, "official test skip present")
    require("pytest exit code: 0" in unit_tests, "pytest exit code mismatch")

    memory = json.loads(
        (results / "memory_evidence.json").read_text(encoding="utf-8")
    )
    require(memory["within_24gib"] is True, "24 GiB evidence failed")
    require(
        memory["pytorch_peak_reserved_mib"] <= 23 * 1024,
        "aggregate peak reserved exceeds allocator limit",
    )
    metadata = json.loads(
        (results / "run_metadata.json").read_text(encoding="utf-8")
    )
    require(
        metadata["formal_process_count"] == 117,
        "run metadata process count mismatch",
    )
    require(
        metadata["minimum_free_memory_mib_at_start"] >= 22 * 1024,
        "run metadata free-memory gate failed",
    )

    asset_files = sorted(path for path in assets.rglob("*") if path.is_file())
    require(len(asset_files) >= 2, "at least two assets are required")
    readme = (assignment / "README.md").read_text(encoding="utf-8")
    for asset in asset_files:
        require(
            asset.relative_to(assignment).as_posix() in readme,
            f"README does not reference {asset.name}",
        )
    require("<填写" not in readme, "README still contains placeholders")

    attachment_bytes = sum(
        path.stat().st_size
        for directory in (results, assets)
        for path in directory.rglob("*")
        if path.is_file()
    )
    require(
        attachment_bytes <= 2 * 1024**2,
        "results and assets exceed 2 MiB",
    )
    print(
        "A2-K artifact validation passed: "
        f"checkpoint=7 attention=18 compile=24 flash=66 "
        f"correctness=38 official_tests=6 attachments={attachment_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
