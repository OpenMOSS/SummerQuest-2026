"""用 local_results 中的正式矩阵重新生成 README 的三张表并就地替换。

与 refresh_a2k_report.py 的区别：本脚本**按章节标题定位**已有的表格块，
而不是依赖 HTML 占位符，因此可以在 README 已经写完表格之后重复执行，
用于保证报告中的数字与 CSV 完全一致。

用法：
  python student_scripts/a2k/sync_readme_tables.py \
      --benchmark local_results/a2k/task5/flash_benchmark.csv \
      --readme ../SummerQuest-2026/students/马万里/assignments/A2-K/README.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PHASES = ("forward", "backward", "forward_backward")
PHASE_LABEL = {"forward": "fwd", "backward": "bwd", "forward_backward": "fwd+bwd"}
IMPLEMENTATIONS = ("eager", "compiled", "triton")


def load(path: Path) -> dict:
    index = {}
    for row in csv.DictReader(path.open(encoding="utf-8")):
        key = (row["implementation"], int(row["sequence_length"]), int(row["head_dim"]), row["phase"])
        index[key] = row
    return index


def speedup(row: dict) -> str:
    value = (row.get("speedup_vs_eager") or "").strip()
    return f"{float(value):.2f}×" if value else "—"


def table(index: dict, sequences: tuple[int, ...], implementations=IMPLEMENTATIONS) -> list[str]:
    lines = [
        "| seq | head_dim | phase | 实现 | p50 (ms) | peak reserved (MiB) | 相对 eager | status |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for seq in sequences:
        for head in (64, 128):
            for phase in PHASES:
                for impl in implementations:
                    row = index.get((impl, seq, head, phase))
                    if row is None:
                        continue
                    lines.append(
                        f"| {seq} | {head} | {PHASE_LABEL[phase]} | {impl} "
                        f"| {float(row['p50_ms']):.4f} | {float(row['peak_reserved_mib']):.0f} "
                        f"| {speedup(row)} | {row['status']} |"
                    )
    return lines


def replace_block(text: str, heading: str, new_lines: list[str]) -> str:
    """把 heading 与其后第一张表之间的内容替换为 new_lines。"""
    start = text.index(heading)
    table_start = text.index("\n|", start) + 1
    table_end = text.index("\n\n", table_start)
    return text[:table_start] + "\n".join(new_lines) + text[table_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()

    index = load(args.benchmark)
    text = args.readme.read_text(encoding="utf-8")

    text = replace_block(text, "### 8.1 核心矩阵", table(index, (512, 2048, 8192)))
    text = replace_block(text, "### 8.2 长序列 16384 边界",
                         table(index, (16384,), ("eager", "triton")))
    args.readme.write_text(text, encoding="utf-8")
    print("已按 CSV 重写 §8.1 与 §8.2 两张表")

    # 自校验：README 中所有形如数据表的行都必须与 CSV 一致
    bad = 0
    checked = 0
    for line in text.splitlines():
        if not line.startswith("| ") or "| ok |" not in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        key = (cells[3], int(cells[0]), int(cells[1]),
               {"fwd": "forward", "bwd": "backward", "fwd+bwd": "forward_backward"}[cells[2]])
        row = index.get(key)
        if row is None:
            print("缺少行:", cells)
            bad += 1
            continue
        checked += 1
        if (abs(float(cells[4]) - float(row["p50_ms"])) > 5e-4
                or abs(float(cells[5]) - float(row["peak_reserved_mib"])) > 0.5):
            print("数值不符:", cells, "| CSV:", row["p50_ms"], row["peak_reserved_mib"])
            bad += 1
    print(f"自校验：{checked} 行与 CSV 一致，{bad} 处不符")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
