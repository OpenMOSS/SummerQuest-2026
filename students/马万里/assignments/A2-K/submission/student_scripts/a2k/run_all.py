"""A2-K 三条正式矩阵的统一公开入口。

本脚本把任务一、任务二、任务五的全部正式实验串成一条可直接复现的命令。
它只调用 `student_scripts/a2k/**` 下已提交的脚本，不依赖任何未提交的编排文件，
也不包含内部调度资源名称——GPU 由调用方以自己环境的方式提供（例如先把本命令
提交到一个只使用单张 RTX 4090 的作业里）。

用法（在 `../assignment2-systems` 仓库根目录）：

    # 四个部分全跑（任务一、任务二、官方 tests 汇总、任务五）
    python student_scripts/a2k/run_all.py --all

    # 只跑其中一个或几个（可组合）
    python student_scripts/a2k/run_all.py --task1 --task5
    python student_scripts/a2k/run_all.py --unit-tests

产物写入 `local_results/a2k/` 与 `local_results/a2k/task5/`。每一条命令在执行前都会
打印出来，便于留档。

所有产物在本地生成后就地脱敏：metadata 里的绝对路径会被改写为仓库内相对路径，
并清除用户名 / 主机名 / IP。**本脚本不做任何复制**——把哪些文件放进提交目录、
如何同步代码，由调用方自行决定（代码同步走
`python3 scripts/sync_a2k_submission.py --name '<同学真名>'`）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path("student_scripts/a2k")
LOCAL = Path("local_results/a2k")
TASK5 = LOCAL / "task5"
FIGURE_DIR = LOCAL / "figures"
STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"

#: 脱敏规则：本地绝对路径会被压成仓库内相对路径；其余常见内部标识替换为占位符。
USER_PATTERNS = (
    (re.compile(r"/remote-home\d+/[^/\s]+"), "<workspace>"),
    (re.compile(r"/home/[^/\s]+"), "<home>"),
    (re.compile(r"/Users/[^/\s]+"), "<home>"),
    (re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b[a-zA-Z0-9._-]+\.(?:local|cluster|internal)\b"), "<host>"),
)


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


def sanitize_text(text: str) -> str:
    """把仓库绝对路径压成相对路径，并替换其余内部标识。

    同时处理 POSIX 与 Windows 两种分隔符：metadata 里记的是 argv，因此在 Windows 上
    会是 `C:\\...\\assignment2-systems\\student_scripts\\...` 这种形式。
    """
    for prefix in (str(REPO_ROOT) + "/", str(REPO_ROOT) + "\\",
                   REPO_ROOT.as_posix() + "/"):
        text = text.replace(prefix, "")
    for pattern, replacement in USER_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_outputs() -> list[Path]:
    """就地脱敏本地产物；返回被改写的文件列表。

    主要处理两类内容：
      * metadata（`*.jsonl` / `*_metadata.json`）——命令里常带脚本的绝对路径；
      * 汇总 JSON——例如 `memory_evidence.json` 里的 `sources[].source`。

    只改写路径与标识，不改动任何测量数字；改写后仍做一次 JSON 合法性兜底校验。
    """
    changed: list[Path] = []
    for path in sorted(LOCAL.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".txt", ".csv"}:
            continue
        raw = path.read_text(encoding="utf-8")
        cleaned = sanitize_text(raw)
        if cleaned != raw:
            path.write_text(cleaned, encoding="utf-8")
            changed.append(path.resolve().relative_to(REPO_ROOT))

    for rel in changed:
        path = REPO_ROOT / rel
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="A2-K 正式矩阵统一入口")
    parser.add_argument("--all", action="store_true", help="跑任务一、二、五与官方 tests")
    parser.add_argument("--task1", action="store_true", help="只跑任务一")
    parser.add_argument("--task2", action="store_true", help="只跑任务二")
    parser.add_argument("--task5", action="store_true", help="只跑任务五")
    parser.add_argument("--unit-tests", action="store_true", help="只跑官方 tests 汇总")
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
        result = subprocess.run(command, check=False, cwd=REPO_ROOT)
        if result.returncode != 0:
            # 单个配置失败不应中止整批：flash_benchmark.py 会把 OOM/失败写进 CSV 行。
            print(f"  注意: {label} 退出码 {result.returncode}", file=sys.stderr, flush=True)

    changed = sanitize_outputs()
    print(f"完成 {len(planned)} 条命令；产物在 {LOCAL}/", flush=True)
    if changed:
        print(f"已就地脱敏 {len(changed)} 个产物文件：", flush=True)
        for rel in changed:
            print(f"  {rel}", flush=True)
    else:
        print("产物已是脱敏状态（无绝对路径或内部标识）。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
