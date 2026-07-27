from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Validate A2-K result coverage without inventing numbers")
    result.add_argument("--results-dir", type=Path, default=Path("results"))
    return result.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_rows(rows: list[dict[str, str]], expected: set[tuple[str, ...]], columns: tuple[str, ...], label: str) -> list[str]:
    actual = {tuple(row[column] for column in columns) for row in rows}
    missing = sorted(expected - actual)
    return [f"{label}: missing {key}" for key in missing]


def main() -> int:
    args = parser()
    root = args.results_dir
    errors: list[str] = []
    required = ("checkpointing.csv", "attention_baseline.csv", "compile_comparison.csv", "flash_benchmark.csv", "correctness.json", "memory_evidence.json", "run_metadata.json")
    for filename in required:
        if not (root / filename).is_file():
            errors.append(f"missing {root / filename}")

    if (root / "checkpointing.csv").is_file():
        rows = read_csv(root / "checkpointing.csv")
        expected = {(str(context), "") for context in (1024, 2048)}
        expected |= {(str(context), str(block)) for context in (1024,) for block in (1, 2, 4, 8)}
        errors += require_rows(rows, expected, ("context_length", "checkpoint_block_size"), "checkpointing")
        if not all(row.get("status") in {"success", "oom", "error"} for row in rows):
            errors.append("checkpointing: unknown status")

    if (root / "attention_baseline.csv").is_file():
        rows = read_csv(root / "attention_baseline.csv")
        expected = {(str(seq), str(dim), phase) for seq in (512, 2048, 8192) for dim in (64, 128) for phase in ("forward", "backward", "forward_backward")}
        errors += require_rows(rows, expected, ("sequence_length", "head_dim", "phase"), "attention_baseline")

    if (root / "compile_comparison.csv").is_file():
        rows = read_csv(root / "compile_comparison.csv")
        expected_attention = {("attention", shape, implementation) for shape in ("seq512-d64", "seq2048-d128", "seq8192-d128") for implementation in ("eager", "compiled")}
        actual_attention = {(row["target"], row["shape"], row["implementation"]) for row in rows}
        errors += [f"compile_comparison: missing {key}" for key in sorted(expected_attention - actual_attention)]
        expected_model = {("model", "small-ctx512", phase, implementation) for phase in ("forward", "forward_backward", "train_step") for implementation in ("eager", "compiled")}
        actual_model = {(row["target"], row["shape"], row["phase"], row["implementation"]) for row in rows}
        errors += [f"compile_comparison: missing {key}" for key in sorted(expected_model - actual_model)]
        if any(row.get("status") == "success" and not row.get("cold_start_ms") for row in rows):
            errors.append("compile_comparison: successful rows need cold_start_ms")

    if (root / "flash_benchmark.csv").is_file():
        rows = read_csv(root / "flash_benchmark.csv")
        expected_core = {(str(seq), str(dim), phase, implementation) for seq in (512, 2048, 8192) for dim in (64, 128) for phase in ("forward", "backward", "forward_backward") for implementation in ("eager", "compiled", "triton")}
        expected_boundary = {("16384", str(dim), phase, implementation) for dim in (64, 128) for phase in ("forward", "backward", "forward_backward") for implementation in ("eager", "triton")}
        errors += require_rows(rows, expected_core | expected_boundary, ("sequence_length", "head_dim", "phase", "implementation"), "flash_benchmark")
        successful_eager = {
            (row["sequence_length"], row["head_dim"], row["dtype"], row["is_causal"], row["phase"])
            for row in rows
            if row["implementation"] == "eager" and row.get("status") == "success"
        }
        for row in rows:
            key = (row["sequence_length"], row["head_dim"], row["dtype"], row["is_causal"], row["phase"])
            if row.get("status") == "success" and row["implementation"] != "eager" and key in successful_eager and not row.get("speedup_vs_eager"):
                errors.append(f"flash_benchmark: missing speedup for {row.get('run_id')}")

    if (root / "correctness.json").is_file():
        data = json.loads((root / "correctness.json").read_text(encoding="utf-8"))
        cases = data.get("cases", [])
        if not cases:
            errors.append("correctness.json: cases is empty")
        if any(case.get("status") == "fail" for case in cases):
            errors.append("correctness.json: at least one case failed")

    if (root / "memory_evidence.json").is_file():
        evidence = json.loads((root / "memory_evidence.json").read_text(encoding="utf-8"))
        if evidence.get("allocator", {}).get("allocator_limit_mib") != 23552:
            errors.append("memory_evidence: allocator limit must be 23552 MiB")
        if evidence.get("hard_limit_mib") != 24576:
            errors.append("memory_evidence: hard limit must be 24576 MiB")
        if float(evidence.get("pytorch_peak_reserved_mib", 0)) > 23552:
            errors.append("memory_evidence: peak reserved exceeds allocator limit")
        if evidence.get("within_24gib") is not True:
            errors.append("memory_evidence: within_24gib must be true")

    if errors:
        print("A2-K result validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"valid A2-K result set: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
