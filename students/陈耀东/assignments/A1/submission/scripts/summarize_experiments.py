"""汇总训练目录中的 config、metrics 和 summary。

输出只使用相对于 runs 根目录的 run 名称，不写入本机或服务器绝对路径。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PUBLIC_CONFIG_KEYS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "normalization",
    "positional_encoding",
    "ffn_type",
    "batch_size",
    "max_iters",
    "max_lr",
    "min_lr",
    "warmup_iters",
    "weight_decay",
    "grad_clip",
    "seed",
    "matmul_precision",
    "compile_mode",
    "amp",
)


def parse_args() -> argparse.Namespace:
    """解析汇总参数。"""
    parser = argparse.ArgumentParser(description="汇总 CS336 实验日志")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    """读取 JSON object；不存在时返回空字典。"""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} 顶层必须是 JSON object")
    return payload


def read_metrics(path: Path) -> list[dict[str, object]]:
    """读取 JSONL 指标并忽略空行。"""
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path.name} 第 {line_number} 行不是 JSON object")
            records.append(record)
    return records


def finite_values(records: list[dict[str, object]], key: str) -> list[float]:
    """提取指定字段中的有限数值。"""
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def has_non_finite_loss(records: list[dict[str, object]]) -> bool:
    """检查 train/val loss 是否出现 NaN 或无穷。"""
    for record in records:
        for key in ("train_loss", "val_loss"):
            value = record.get(key)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                return True
    return False


def discover_run_directories(root: Path) -> list[Path]:
    """发现含 config、metrics 或 summary 的实验目录。"""
    run_directories: set[Path] = set()
    for filename in ("config.json", "metrics.jsonl", "summary.json"):
        run_directories.update(path.parent for path in root.rglob(filename))
    return sorted(run_directories)


def summarize_run(root: Path, run_directory: Path) -> dict[str, object]:
    """汇总一个 run，不泄露绝对路径。"""
    config = read_json(run_directory / "config.json")
    summary = read_json(run_directory / "summary.json")
    metrics = read_metrics(run_directory / "metrics.jsonl")
    train_losses = finite_values(metrics, "train_loss")
    val_losses = finite_values(metrics, "val_loss")

    status = str(summary.get("status", "completed")) if summary else "incomplete"
    if has_non_finite_loss(metrics):
        status = "diverged"
    if any(record.get("failure") for record in metrics):
        status = "diverged"

    public_config = {key: config[key] for key in PUBLIC_CONFIG_KEYS if key in config}
    last_record = metrics[-1] if metrics else {}
    return {
        "run": run_directory.relative_to(root).as_posix(),
        "status": status,
        "metric_points": len(metrics),
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "final_val_loss": summary.get("final_val_loss"),
        "last_step": last_record.get("step"),
        "processed_tokens": summary.get("processed_tokens", last_record.get("processed_tokens")),
        "total_wall_clock_sec": summary.get("total_wall_clock_sec"),
        "parameter_count": summary.get("parameter_count"),
        "average_step_time_sec": summary.get("average_step_time_sec"),
        "average_tokens_per_second": summary.get("average_tokens_per_second"),
        "cuda_peak_memory_bytes": summary.get("cuda_peak_memory_bytes"),
        "failure": summary.get("failure"),
        "config": public_config,
    }


def main() -> None:
    """汇总所有实验并写出稳定 JSON。"""
    args = parse_args()
    root = args.runs_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"runs 根目录不存在：{args.runs_root}")

    output_path = args.output or args.runs_root / "experiment_summary.json"
    payload = {"runs": [summarize_run(root, run_directory) for run_directory in discover_run_directories(root)]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"runs={len(payload['runs'])} output={output_path}")


if __name__ == "__main__":
    main()
