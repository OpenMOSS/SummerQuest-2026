"""A2-K 三条正式矩阵的统一公开入口。

本脚本把任务一、任务二、任务五的全部正式实验串成一条可直接复现的命令。
它只调用 `student_scripts/a2k/**` 下已提交的脚本，不依赖任何未提交的编排文件，
也不包含内部调度资源名称——GPU 由调用方以自己环境的方式提供（例如先把本命令
提交到一个只使用单张 RTX 4090 的作业里）。

用法（在 `../assignment2-systems` 仓库根目录）：

    # 三个任务全跑
    python student_scripts/a2k/run_all.py --all

    # 只跑某一个任务（可组合）
    python student_scripts/a2k/run_all.py --task1 --task5

    # 只打印将要执行的命令，不执行
    python student_scripts/a2k/run_all.py --all --dry-run

产物写入 `local_results/a2k/` 与 `local_results/a2k/task5/`；把其中的轻量汇总与图片
脱敏后放入提交目录的 `results/` 与 `assets/`。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path("student_scripts/a2k")
LOCAL = Path("local_results/a2k")
TASK5 = LOCAL / "task5"
FIGURE_DIR = LOCAL / "figures"
STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"


def task1_commands() -> list[list[str]]:
    """任务一：1024 标准矩阵（none + block 1/2/4/8）与 2048 边界，一条命令跑完。"""
    return [[
        sys.executable, str(SCRIPTS / "checkpointing_benchmark.py"),
        "--batch",
        "--context-lengths", "1024", "2048",
        "--block-sizes", "1", "2", "4", "8",
        "--warmup", "3", "--steps", "5", "--seed", "42",
        "--output", str(LOCAL / "checkpointing.csv"),
    ]]


def task2_commands() -> list[list[str]]:
    """任务二：baseline 与 compile 两个矩阵，`--batch` 内部按配置各起一个子进程。"""
    common = ["--batch", "--warmup-ms", "100", "--rep-ms", "300", "--seed", "42"]
    return [
        [
            sys.executable, str(SCRIPTS / "attention_benchmark.py"), *common,
            "--matrix", "baseline",
            "--output", str(LOCAL / "attention_baseline.csv"),
            "--metadata-output", str(LOCAL / "attention_baseline_metadata.jsonl"),
        ],
        [
            sys.executable, str(SCRIPTS / "attention_benchmark.py"), *common,
            "--matrix", "compile",
            "--output", str(LOCAL / "compile_comparison.csv"),
            "--metadata-output", str(LOCAL / "compile_comparison_metadata.jsonl"),
        ],
    ]


def task5_commands() -> list[list[str]]:
    """任务五：环境 metadata、扩展正确性、性能矩阵、汇总与绘图。

    性能矩阵用 `flash_benchmark.py --batch`，它内部为 72 个配置各起一个独立进程。
    """
    return [
        [
            sys.executable, str(SCRIPTS / "task5_metadata.py"),
            "--output", str(TASK5 / "run_metadata.json"),
            "--commit", STARTER_COMMIT,
            "--seed", "42",
        ],
        [
            sys.executable, str(SCRIPTS / "task5_correctness.py"),
            "--output", str(TASK5 / "correctness.json"),
            "--length", "128", "512", "2048",
        ],
        [
            sys.executable, str(SCRIPTS / "flash_benchmark.py"),
            "--batch",
            "--output", str(TASK5 / "flash_benchmark.csv"),
        ],
        [
            sys.executable, str(SCRIPTS / "summarize_flash_benchmark.py"),
            "--input", str(TASK5 / "flash_benchmark.csv"),
        ],
        [
            sys.executable, str(SCRIPTS / "summarize_memory_evidence.py"),
            "--output", str(LOCAL / "memory_evidence.json"),
        ],
        [
            sys.executable, str(SCRIPTS / "plot_task5.py"),
            "--output-dir", str(FIGURE_DIR),
            "--benchmark", str(TASK5 / "flash_benchmark.csv"),
            "--checkpointing", str(LOCAL / "checkpointing.csv"),
        ],
    ]


def unit_test_command() -> list[str]:
    """官方 tests：由 summarize_flash_tests.py 执行 pytest 并把输出脱敏成报告。"""
    return [
        sys.executable, str(SCRIPTS / "summarize_flash_tests.py"),
        "--run", f"{sys.executable} -m pytest tests/test_attention.py -v",
        "--gpu", "NVIDIA GeForce RTX 4090",
        "--commit", STARTER_COMMIT,
        "--output-dir", str(LOCAL / "pytest"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="A2-K 正式矩阵统一入口")
    parser.add_argument("--all", action="store_true", help="跑任务一、二、五与官方 tests")
    parser.add_argument("--task1", action="store_true", help="只跑任务一")
    parser.add_argument("--task2", action="store_true", help="只跑任务二")
    parser.add_argument("--task5", action="store_true", help="只跑任务五")
    parser.add_argument("--unit-tests", action="store_true", help="只跑官方 tests 汇总")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    args = parser.parse_args()

    selected = {
        "task1": args.all or args.task1,
        "task2": args.all or args.task2,
        "task5": args.all or args.task5,
        "unit_tests": args.all or args.unit_tests,
    }
    if not any(selected.values()):
        parser.error("请指定 --all 或至少一个任务开关")

    planned: list[tuple[str, list[str]]] = []
    if selected["task1"]:
        planned += [("任务一 checkpointing", c) for c in task1_commands()]
    if selected["task2"]:
        planned += [("任务二 attention", c) for c in task2_commands()]
    if selected["unit_tests"]:
        planned.append(("官方 tests 汇总", unit_test_command()))
    if selected["task5"]:
        planned += [("任务五 性能矩阵", c) for c in task5_commands()]

    for label, command in planned:
        print(f"[{label}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, check=False, cwd=REPO_ROOT)
        if result.returncode != 0:
            # 单个配置失败不应中止整批：flash_benchmark.py 会把 OOM/失败写进 CSV 行。
            print(f"  注意: {label} 退出码 {result.returncode}", file=sys.stderr, flush=True)

    if args.dry_run:
        print(f"共 {len(planned)} 条命令（--dry-run 未执行）")
    else:
        print(f"完成 {len(planned)} 条命令；产物在 {LOCAL}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
