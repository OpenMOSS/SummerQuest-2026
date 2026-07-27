"""把三项任务的 JSONL 原始结果整理成 CSV 与 Markdown 表。

矩阵 runner 负责保留尽可能完整的原始记录，本脚本只生成便于阅读和写报告的
扁平视图，不修改原数据。不同 event 会展开为不同表，OOM 和 process error
同样保留，避免制表时无声删除失败配置。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    """按输入顺序读取多个 JSONL；损坏行直接报出文件和行号。"""

    records: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return records


def get(record: dict[str, Any], *path: str) -> Any:
    """读取嵌套字段；OOM 等缺字段记录返回空值。"""

    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def benchmark_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "model_size": record.get("model_size"),
        "context_length": get(record, "model", "context_length"),
        "batch_size": record.get("batch_size"),
        "mode": record.get("mode"),
        "precision": record.get("precision"),
        "compiled": record.get("compile_model"),
        "compile_cold_start_ms": record.get("compile_cold_start_ms"),
        "warmup": record.get("warmup"),
        "repeats": record.get("repeats"),
        "mean_ms": get(record, "timing", "mean_ms"),
        "std_ms": get(record, "timing", "std_ms"),
        "cv": get(record, "timing", "cv"),
        "p20_ms": get(record, "timing", "p20_ms"),
        "p50_ms": get(record, "timing", "p50_ms"),
        "p80_ms": get(record, "timing", "p80_ms"),
        "measurement_count": get(record, "timing", "measurement_count"),
        "peak_allocated_mib": get(record, "peak_memory", "max_allocated_mib"),
        "peak_reserved_mib": get(record, "peak_memory", "max_reserved_mib"),
        "gpu": get(record, "environment", "gpu_name"),
        "error": record.get("error") or record.get("stderr_tail"),
    }


def attention_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "implementation": record.get("implementation"),
        "dtype": record.get("dtype"),
        "batch_size": record.get("batch_size"),
        "causal": record.get("causal"),
        "sequence_length": record.get("sequence_length"),
        "head_dimension": record.get("head_dimension"),
        "compile_cold_start_ms": record.get("compile_cold_start_ms"),
        "forward_status": get(record, "forward", "status"),
        "forward_mean_ms": get(record, "forward", "mean_ms"),
        "forward_std_ms": get(record, "forward", "std_ms"),
        "forward_p20_ms": get(record, "forward", "p20_ms"),
        "forward_p50_ms": get(record, "forward", "p50_ms"),
        "forward_p80_ms": get(record, "forward", "p80_ms"),
        "backward_status": get(record, "backward", "status"),
        "backward_mean_ms": get(record, "backward", "mean_ms"),
        "backward_std_ms": get(record, "backward", "std_ms"),
        "backward_p20_ms": get(record, "backward", "p20_ms"),
        "backward_p50_ms": get(record, "backward", "p50_ms"),
        "backward_p80_ms": get(record, "backward", "p80_ms"),
        "forward_backward_status": get(record, "forward_backward", "status"),
        "forward_backward_mean_ms": get(record, "forward_backward", "mean_ms"),
        "forward_backward_p20_ms": get(record, "forward_backward", "p20_ms"),
        "forward_backward_p50_ms": get(record, "forward_backward", "p50_ms"),
        "forward_backward_p80_ms": get(record, "forward_backward", "p80_ms"),
        "memory_before_backward_mib": get(record, "memory_before_backward", "allocated_mib"),
        "forward_peak_allocated_mib": get(record, "forward", "memory", "peak_allocated_mib"),
        "backward_peak_allocated_mib": get(record, "backward", "memory", "peak_allocated_mib"),
        "forward_backward_peak_allocated_mib": get(record, "forward_backward", "memory", "peak_allocated_mib"),
        "forward_peak_reserved_mib": get(record, "forward", "memory", "peak_reserved_mib"),
        "backward_peak_reserved_mib": get(record, "backward", "memory", "peak_reserved_mib"),
        "forward_backward_peak_reserved_mib": get(record, "forward_backward", "memory", "peak_reserved_mib"),
        "scores_plus_probabilities_mib": get(record, "theoretical_memory", "scores_plus_probabilities_mib"),
        "gpu": get(record, "environment", "gpu_name"),
        "error": record.get("error") or record.get("stderr_tail"),
    }


def flash_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "implementation": record.get("implementation"),
        "dtype": record.get("dtype"),
        "causal": record.get("causal"),
        "sequence_length": record.get("sequence_length"),
        "head_dimension": record.get("head_dimension"),
        "compile_cold_start_ms": record.get("compile_cold_start_ms"),
        "forward_status": get(record, "forward", "status"),
        "forward_mean_ms": get(record, "forward", "mean_ms"),
        "forward_p20_ms": get(record, "forward", "p20_ms"),
        "forward_p50_ms": get(record, "forward", "p50_ms"),
        "forward_p80_ms": get(record, "forward", "p80_ms"),
        "backward_status": get(record, "backward", "status"),
        "backward_mean_ms": get(record, "backward", "mean_ms"),
        "backward_p20_ms": get(record, "backward", "p20_ms"),
        "backward_p50_ms": get(record, "backward", "p50_ms"),
        "backward_p80_ms": get(record, "backward", "p80_ms"),
        "forward_backward_status": get(record, "forward_backward", "status"),
        "forward_backward_mean_ms": get(record, "forward_backward", "mean_ms"),
        "forward_backward_p20_ms": get(record, "forward_backward", "p20_ms"),
        "forward_backward_p50_ms": get(record, "forward_backward", "p50_ms"),
        "forward_backward_p80_ms": get(record, "forward_backward", "p80_ms"),
        "end_to_end_status": get(record, "end_to_end", "status"),
        "end_to_end_mean_ms": get(record, "end_to_end", "mean_ms"),
        "end_to_end_p20_ms": get(record, "end_to_end", "p20_ms"),
        "end_to_end_p50_ms": get(record, "end_to_end", "p50_ms"),
        "end_to_end_p80_ms": get(record, "end_to_end", "p80_ms"),
        "memory_before_backward_mib": record.get("memory_before_backward_mib"),
        "forward_peak_allocated_mib": get(record, "forward", "memory", "peak_allocated_mib"),
        "backward_peak_allocated_mib": get(record, "backward", "memory", "peak_allocated_mib"),
        "forward_backward_peak_allocated_mib": get(record, "forward_backward", "memory", "peak_allocated_mib"),
        "forward_peak_reserved_mib": get(record, "forward", "memory", "peak_reserved_mib"),
        "backward_peak_reserved_mib": get(record, "backward", "memory", "peak_reserved_mib"),
        "forward_backward_peak_reserved_mib": get(record, "forward_backward", "memory", "peak_reserved_mib"),
        "triton_query_tile": get(record, "environment", "triton_config", "query_tile"),
        "triton_key_tile": get(record, "environment", "triton_config", "key_tile"),
        "triton_num_warps": get(record, "environment", "triton_config", "num_warps"),
        "triton_num_stages": get(record, "environment", "triton_config", "num_stages"),
        "gpu": get(record, "environment", "gpu_name"),
        "error": record.get("error") or record.get("stderr_tail"),
    }


def memory_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": record.get("case_id"),
        "event": record.get("event"),
        "status": record.get("status"),
        "component": record.get("component"),
        "compiled": record.get("compiled") if "compiled" in record else record.get("compiled_blocks"),
        "batch_size": record.get("batch_size"),
        "sequence_length": record.get("sequence_length"),
        "num_layers": record.get("num_layers"),
        "checkpoint_strategy": record.get("checkpoint_strategy"),
        "checkpoint_group_size": record.get("checkpoint_group_size"),
        "saved_unique_storage_mib": get(record, "summary", "unique_non_parameter_storage_mib"),
        "forward_ms": get(record, "timing_ms", "forward"),
        "backward_ms": get(record, "timing_ms", "backward"),
        "peak_allocated_mib": get(record, "memory", "after_backward", "peak_allocated_mib"),
        "peak_increment_mib": get(record, "memory", "peak_increment_over_forward_baseline_mib"),
        "output_sum": get(record, "output_summary", "sum"),
        "input_gradient_norm": get(record, "input_gradient_summary", "norm"),
        "error": record.get("error") or record.get("stderr_tail"),
    }


def classify(record: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """按 event 选择扁平 schema。"""

    event = record.get("event")
    if event == "benchmark_training_step":
        return "benchmark", benchmark_row(record)
    if event == "pytorch_attention_benchmark":
        return "attention", attention_row(record)
    if event == "flashattention_benchmark":
        return "flash", flash_row(record)
    if event in ("saved_tensors_experiment", "checkpoint_experiment"):
        return "memory", memory_row(record)
    return "other", {"event": event, "status": record.get("status"), "record": json.dumps(record, ensure_ascii=False)}


def format_cell(value: Any) -> str:
    """CSV 保留原精度，Markdown 对浮点数做紧凑显示。"""

    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value).replace("\n", " ")
    return text


def write_table(name: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    """同一份扁平数据同时写 CSV 与 Markdown。"""

    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    csv_path = output_dir / f"{name}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / f"{name}.md"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(field)).replace("|", "\\|") for field in fields) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert A2 JSONL results to CSV/Markdown tables.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[dict[str, Any]]] = {}
    for record in read_jsonl(args.inputs):
        name, row = classify(record)
        tables.setdefault(name, []).append(row)
    for name, rows in tables.items():
        write_table(name, rows, args.output_dir)
    (args.output_dir / "summary.json").write_text(
        json.dumps({name: len(rows) for name, rows in tables.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({name: len(rows) for name, rows in tables.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
