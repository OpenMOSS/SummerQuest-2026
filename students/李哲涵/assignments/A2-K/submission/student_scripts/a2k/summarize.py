from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from cs336_systems.a2k.runtime import write_json


def read_json_files(directory: Path) -> list[dict[str, Any]]:
    rows = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serializable = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serializable)


def add_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eager = {
        (
            row.get("seq_len"),
            row.get("head_dim"),
            row.get("phase"),
            row.get("dtype"),
            row.get("causal"),
        ): row
        for row in rows
        if row.get("implementation") == "torch_eager"
        and row.get("status") == "success"
    }
    enriched = []
    for row in rows:
        row = dict(row)
        baseline = eager.get(
            (
                row.get("seq_len"),
                row.get("head_dim"),
                row.get("phase"),
                row.get("dtype"),
                row.get("causal"),
            )
        )
        if (
            baseline is not None
            and row.get("status") == "success"
            and row.get("p50_ms")
        ):
            row["speedup_vs_eager"] = (
                baseline["p50_ms"] / row["p50_ms"]
            )
        else:
            row["speedup_vs_eager"] = None
        enriched.append(row)
    return enriched


def summarize(root: Path) -> None:
    raw = root / "raw"
    results = root / "results"
    correctness_path = results / "correctness.json"
    correctness_rows = (
        [json.loads(correctness_path.read_text())]
        if correctness_path.exists()
        else []
    )
    checkpoint_rows = read_json_files(raw / "checkpoint")
    attention_rows = add_speedups(read_json_files(raw / "attention"))
    model_compile_rows = read_json_files(raw / "compile_model")

    checkpoint_fields = [
        "config_id",
        "model_size",
        "num_layers",
        "context_length",
        "batch_size",
        "dtype",
        "checkpoint_block_size",
        "nested",
        "warmup_steps",
        "measurement_steps",
        "step_time_ms_samples",
        "step_time_ms_p50",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "status",
        "error_type",
        "error_message",
    ]
    write_csv(
        results / "checkpointing.csv",
        checkpoint_rows,
        checkpoint_fields,
    )

    attention_fields = [
        "implementation",
        "batch_size",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "phase",
        "warmup_ms",
        "measurement_ms",
        "p20_ms",
        "p50_ms",
        "p80_ms",
        "cold_start_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "speedup_vs_eager",
        "block_q",
        "block_k",
        "num_warps",
        "num_stages",
        "status",
        "error_type",
        "error_message",
    ]
    baseline_rows = [
        row
        for row in attention_rows
        if row.get("implementation") == "torch_eager"
        and row.get("seq_len") in (512, 2048, 8192)
        and row.get("head_dim") in (64, 128)
    ]
    write_csv(
        results / "attention_baseline.csv",
        baseline_rows,
        attention_fields,
    )
    write_csv(
        results / "flash_benchmark.csv",
        attention_rows,
        attention_fields,
    )

    compile_attention_shapes = {(512, 64), (2048, 128), (8192, 128)}
    compile_rows = [
        {
            "scope": "attention",
            **row,
        }
        for row in attention_rows
        if row.get("implementation") in ("torch_eager", "torch_compiled")
        and (row.get("seq_len"), row.get("head_dim"))
        in compile_attention_shapes
    ]
    compile_rows.extend(model_compile_rows)
    compile_fields = [
        "scope",
        "implementation",
        "model_size",
        "batch_size",
        "context_length",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "mode",
        "phase",
        "warmup_steps",
        "measurement_steps",
        "warmup_ms",
        "measurement_ms",
        "cold_start_ms",
        "samples_ms",
        "p20_ms",
        "p50_ms",
        "p80_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "status",
        "error_type",
        "error_message",
    ]
    write_csv(
        results / "compile_comparison.csv",
        compile_rows,
        compile_fields,
    )

    all_rows = (
        correctness_rows
        + checkpoint_rows
        + attention_rows
        + model_compile_rows
    )
    performance_rows = checkpoint_rows + attention_rows + model_compile_rows
    successful = [row for row in all_rows if row.get("status") == "success"]
    peak_allocated = max(
        (row.get("peak_allocated_mib", 0.0) for row in successful),
        default=0.0,
    )
    peak_reserved = max(
        (row.get("peak_reserved_mib", 0.0) for row in successful),
        default=0.0,
    )
    process_metadata = [
        row["metadata"]
        for row in all_rows
        if isinstance(row.get("metadata"), dict)
    ]
    performance_metadata = next(
        (
            row["metadata"]
            for row in performance_rows
            if isinstance(row.get("metadata"), dict)
        ),
        {},
    )
    correctness_metadata = next(
        (
            row["metadata"]
            for row in correctness_rows
            if isinstance(row.get("metadata"), dict)
        ),
        {},
    )
    first_metadata = performance_metadata or correctness_metadata
    start_free_values = [
        metadata["gpu_free_memory_at_start_mib"]
        for metadata in process_metadata
        if isinstance(metadata.get("gpu_free_memory_at_start_mib"), (int, float))
    ]
    memory_evidence = {
        "allocator": {
            "allocator_fraction": first_metadata.get("allocator_fraction"),
            "allocator_limit_mib": first_metadata.get("allocator_limit_mib", 23552),
        },
        "hard_limit_mib": first_metadata.get("hard_limit_mib", 24576),
        "pytorch_peak_allocated_mib": peak_allocated,
        "pytorch_peak_reserved_mib": peak_reserved,
        "within_24gib": peak_reserved <= 23552,
        "minimum_gpu_free_memory_at_start_mib": (
            min(start_free_values) if start_free_values else None
        ),
        "successful_processes": len(successful),
        "failed_or_oom_processes": len(all_rows) - len(successful),
    }
    write_json(results / "memory_evidence.json", memory_evidence)

    performance_revision = next(
        (
            row.get("commit")
            for row in performance_rows
            if row.get("status") == "success" and row.get("commit")
        ),
        None,
    )
    correctness_revision = next(
        (
            row.get("commit")
            for row in correctness_rows
            if row.get("status") == "success" and row.get("commit")
        ),
        None,
    )
    source_revisions = sorted(
        {
            row["commit"]
            for row in all_rows
            if isinstance(row.get("commit"), str)
        }
    )
    run_metadata = {
        "commit": performance_revision or correctness_revision,
        "correctness_revision": correctness_revision,
        "source_revisions": source_revisions,
        "seed": 0,
        "hardware_software": first_metadata,
        "correctness_settings": {
            "tf32_matmul_allowed": correctness_metadata.get(
                "tf32_matmul_allowed"
            ),
            "tf32_cudnn_allowed": correctness_metadata.get(
                "tf32_cudnn_allowed"
            ),
            "includes_fp32": True,
        },
        "process_metadata_summary": {
            "recorded_processes": len(process_metadata),
            "minimum_gpu_free_memory_at_start_mib": (
                min(start_free_values) if start_free_values else None
            ),
        },
        "formal_protocol": {
            "gpu_processes": "serial independent processes",
            "allocator_limit_mib": 23552,
            "minimum_free_memory_mib": 22528,
            "attention_warmup_ms": 100,
            "attention_measurement_ms": 300,
            "attention_quantiles": [0.2, 0.5, 0.8],
            "performance_dtype": "bf16",
            "attention_timer": "triton.testing.do_bench using CUDA events",
            "model_timer": (
                "time.perf_counter bracketed by torch.cuda.synchronize"
            ),
        },
        "row_counts": {
            "checkpoint": len(checkpoint_rows),
            "attention": len(attention_rows),
            "compile_model": len(model_compile_rows),
            "correctness": len(correctness_rows),
        },
        "commands": [
            row.get("command")
            for row in all_rows
            if isinstance(row.get("command"), list)
        ],
    }
    write_json(results / "run_metadata.json", run_metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="local_results/a2k")
    args = parser.parse_args()
    summarize(Path(args.root))


if __name__ == "__main__":
    main()
