"""A2-K Activation Checkpointing 正式矩阵运行器。

固定题目要求的训练配置：Stanford medium、24 层、batch=1、context=1024、
FP32 参数、BF16 autocast、AdamW。第一阶段扫描无 checkpoint 和 group size
1/2/4/8；第二阶段在 context=2048 上运行无 checkpoint与第一阶段的最佳
checkpoint 配置。每个 case 都由独立 Python 进程执行，OOM 会保留在 CSV 中。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
GROUP_SIZES = (0, 1, 2, 4, 8)


@dataclass(frozen=True)
class CheckpointCase:
    """一个可独立运行的正式 checkpoint case。"""

    context_length: int
    group_size: int
    label: str

    @property
    def case_id(self) -> str:
        return f"medium-l24-b1-s{self.context_length}-g{self.group_size}-{self.label}"


def first_stage_cases() -> list[CheckpointCase]:
    return [CheckpointCase(1024, group_size, "scan") for group_size in GROUP_SIZES]


def read_last_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def case_command(case: CheckpointCase, output: Path, *, python: str, warmup: int, repeats: int) -> list[str]:
    """生成固定 shape 的子进程命令，避免 shell 拼接造成参数歧义。"""

    return [
        python,
        "-m",
        "cs336_systems.a2k.memory_experiments",
        "checkpoint",
        "--device",
        "cuda",
        "--batch-size",
        "1",
        "--sequence-length",
        str(case.context_length),
        "--d-model",
        "1024",
        "--d-ff",
        "4096",
        "--num-heads",
        "16",
        "--num-layers",
        "24",
        "--checkpoint-strategy",
        "none" if case.group_size == 0 else "groups",
        "--checkpoint-group-size",
        str(case.group_size),
        "--precision",
        "bf16-mixed",
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--allocator-limit-gib",
        "23",
        "--allow-oom",
        "--output",
        str(output),
    ]


def run_case(case: CheckpointCase, run_dir: Path, *, python: str, warmup: int, repeats: int) -> dict[str, Any]:
    """运行单个配置并保留命令与 stdout/stderr。"""

    case_dir = run_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.jsonl"
    command = case_command(case, result_path, python=python, warmup=warmup, repeats=repeats)
    (case_dir / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    result = read_last_json(result_path)
    if result is None:
        result = {
            "event": "checkpoint_experiment",
            "status": "process_error",
            "error": completed.stderr[-4000:],
        }
    result.update(case_id=case.case_id, context_length=case.context_length, group_size=case.group_size)
    return result


def choose_best(records: list[dict[str, Any]]) -> int | None:
    """按正式主指标选择 checkpoint group size。

    ``peak_allocated_mib`` 表示活跃 tensor 的真实峰值，是 checkpoint
    策略比较的主指标；相对 forward 基线的增量只作为同峰值时的辅助排序。
    ``peak_reserved_mib`` 包含 allocator 缓存，不能直接作为最佳策略依据。
    """

    candidates: list[tuple[float, float, int]] = []
    for record in records:
        if record.get("status") != "passed":
            continue
        memory = record.get("memory") or {}
        peak_allocated = memory.get("peak_allocated_mib")
        if peak_allocated is None:
            continue
        peak_increment = memory.get("peak_increment_over_forward_baseline_mib")
        candidates.append(
            (
                float(peak_allocated),
                float(peak_increment) if peak_increment is not None else float("inf"),
                int(record["group_size"]),
            )
        )
    return min(candidates)[2] if candidates else None


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    timing = record.get("timing_ms") or {}
    train_timing = timing.get("train_step") or {}
    memory = record.get("memory") or {}
    environment = record.get("environment") or {}
    return {
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "context_length": record.get("context_length", record.get("sequence_length")),
        "batch_size": record.get("batch_size"),
        "d_model": record.get("d_model"),
        "num_layers": record.get("num_layers"),
        "precision": record.get("precision"),
        "parameter_dtype": record.get("parameter_dtype"),
        "optimizer": record.get("optimizer"),
        "checkpoint_strategy": record.get("checkpoint_strategy"),
        "group_size": record.get("group_size", record.get("checkpoint_group_size")),
        "warmup": record.get("warmup"),
        "repeats": record.get("repeats"),
        "mean_ms": train_timing.get("mean_ms"),
        "std_ms": train_timing.get("std_ms"),
        "cv": train_timing.get("cv"),
        "p20_ms": train_timing.get("p20_ms"),
        "p50_ms": train_timing.get("p50_ms"),
        "p80_ms": train_timing.get("p80_ms"),
        "peak_allocated_mib": memory.get("peak_allocated_mib"),
        "peak_reserved_mib": memory.get("peak_reserved_mib"),
        "peak_increment_mib": memory.get("peak_increment_over_forward_baseline_mib"),
        "gpu": environment.get("gpu_name"),
        "error": record.get("error"),
    }


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = [flatten(record) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["case_id", "status"]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the formal A2-K activation-checkpointing matrix.")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")

    run_dir = args.run_dir or REPO_ROOT / "artifacts" / "a2k-checkpointing"
    # formal runner 会预先创建子目录保存命令和日志，因此首次运行也必须接受
    # 已存在的 run_dir；续跑判断仍由 --resume 和已有 result 文件完成。
    run_dir.mkdir(parents=True, exist_ok=True)
    scan_cases = first_stage_cases()
    manifest = {
        "assignment": "A2-K",
        "task": "activation_checkpointing",
        "fixed_configuration": {
            "model_size": "medium",
            "num_layers": 24,
            "batch_size": 1,
            "context_length": 1024,
            "parameter_dtype": "torch.float32",
            "compute_dtype": "torch.bfloat16 autocast",
            "optimizer": "AdamW",
            "allocator_limit_gib": 23,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "scan_cases": [{"case_id": case.case_id, **asdict(case)} for case in scan_cases],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    results_path = run_dir / "results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["case_id"]] = record

    records: list[dict[str, Any]] = []
    if not args.dry_run:
        for case in scan_cases:
            record = existing.get(case.case_id) or run_case(
                case, run_dir, python=args.python, warmup=args.warmup, repeats=args.repeats
            )
            if case.case_id not in existing:
                with results_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

        best_group_size = choose_best(records)
        follow_up = [CheckpointCase(2048, 0, "baseline")]
        if best_group_size is not None and best_group_size != 0:
            follow_up.append(CheckpointCase(2048, best_group_size, "best"))
        manifest["best_group_size_at_1024"] = best_group_size
        manifest["follow_up_cases"] = [{"case_id": case.case_id, **asdict(case)} for case in follow_up]
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for case in follow_up:
            record = existing.get(case.case_id) or run_case(
                case, run_dir, python=args.python, warmup=args.warmup, repeats=args.repeats
            )
            if case.case_id not in existing:
                with results_path.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    write_csv(records, run_dir / "checkpointing.csv")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "assignment": "A2-K",
                "task": "activation_checkpointing",
                "created_at": datetime.now().astimezone().isoformat(),
                "python": args.python,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "dry_run": args.dry_run,
                "case_count": len(records),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
