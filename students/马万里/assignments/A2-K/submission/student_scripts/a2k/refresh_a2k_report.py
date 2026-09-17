"""把正式矩阵的产物收尾进 A2-K 提交目录，并刷新 README 中的性能表格。

步骤：
  1. 读取 local_results 中的正式矩阵（checkpointing / attention / compile / task5）；
  2. 生成核心矩阵、16384 边界、失败行三张 Markdown 表；
  3. 用三张表替换 README.md 里的 HTML 注释占位符；
  4. 把 8 个必需结果文件与 4 张图平铺拷贝进提交目录（assets/ 与 results/）；
  5. 脱敏并固定 *metadata.jsonl 中的 commit；
  6. 校验附件体积上限（results + assets <= 2 MiB）。

用法：
  python student_scripts/a2k/refresh_a2k_report.py \
      --local-root local_results/a2k \
      --assignment ../SummerQuest-2026/students/马万里/assignments/A2-K
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

REQUIRED_RESULTS = (
    "correctness.json",
    "unit_tests.txt",
    "checkpointing.csv",
    "attention_baseline.csv",
    "compile_comparison.csv",
    "flash_benchmark.csv",
    "memory_evidence.json",
    "run_metadata.json",
)
FIGURES = (
    "task1_checkpointing_tradeoff.png",
    "task5_latency.png",
    "task5_speedup.png",
    "task5_memory.png",
)
IMPLEMENTATIONS = ("eager", "compiled", "triton")
PHASES = ("forward", "backward", "forward_backward")
PHASE_LABEL = {"forward": "fwd", "backward": "bwd", "forward_backward": "fwd+bwd"}
MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024

# local_results 中的源文件 -> 提交 results/ 里的目标名
SOURCE_MAP = {
    "task5/correctness.json": "correctness.json",
    "task5/flash_benchmark.csv": "flash_benchmark.csv",
    "task5/run_metadata.json": "run_metadata.json",
    "memory_evidence.json": "memory_evidence.json",
    "checkpointing.csv": "checkpointing.csv",
    "attention_baseline.csv": "attention_baseline.csv",
    "compile_comparison.csv": "compile_comparison.csv",
    "pytest/unit_tests.txt": "unit_tests.txt",
}


def load_matrix(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        row["_seq"] = int(row["sequence_length"])
        row["_head"] = int(row["head_dim"])
    return rows


def speedup_text(row: dict) -> str:
    value = (row.get("speedup_vs_eager") or "").strip()
    return f"{float(value):.2f}×" if value else "—"


def status_text(row: dict) -> str:
    status = (row.get("status") or "").strip() or "?"
    if status != "ok":
        error = (row.get("error") or "").strip().replace("|", "/")
        return f"**{status}**（{error[:60]}）" if error else f"**{status}**"
    return "ok"


def matrix_table(rows: list[dict], sequences: tuple[int, ...]) -> str:
    lines = [
        "| seq | head_dim | phase | 实现 | p50 (ms) | peak reserved (MiB) | 相对 eager | status |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    index = {(r["implementation"], r["_seq"], r["_head"], r["phase"]): r for r in rows}
    for seq in sequences:
        for head in (64, 128):
            for phase in PHASES:
                for impl in IMPLEMENTATIONS:
                    row = index.get((impl, seq, head, phase))
                    if row is None:
                        continue
                    lines.append(
                        f"| {seq} | {head} | {PHASE_LABEL[phase]} | {impl} "
                        f"| {float(row['p50_ms']):.4f} | {float(row['peak_reserved_mib']):.0f} "
                        f"| {speedup_text(row)} | {status_text(row)} |"
                    )
    return "\n".join(lines)


def boundary_table(rows: list[dict]) -> str:
    wanted = [r for r in rows if r["_seq"] == 16384 and r["implementation"] != "compiled"]
    if not wanted:
        return "（本次运行未产生 16384 边界行）"
    return matrix_table(wanted, (16384,))


def failure_table(rows: list[dict], correctness_path: Path) -> str:
    bad = [r for r in rows if (r.get("status") or "") != "ok"]
    total = len(rows)
    ok = total - len(bad)
    correctness = json.loads(correctness_path.read_text(encoding="utf-8")) if correctness_path.is_file() else []
    n_pass = sum(1 for r in correctness if r.get("status") == "pass")
    lines = [
        f"- 性能矩阵共 **{total} 行**，`status == ok` 的 **{ok} 行**"
        + (f"，非 ok 的 **{len(bad)} 行**：" if bad else "，**无 OOM、无编译失败**。"),
        f"- 扩展正确性共 **{len(correctness)} 项**，`status == pass` 的 **{n_pass} 项**。",
    ]
    if bad:
        lines.append("")
        lines.append("| seq | head_dim | phase | 实现 | status | error |")
        lines.append("| ---: | ---: | --- | --- | --- | --- |")
        for row in sorted(bad, key=lambda r: (r["_seq"], r["_head"], r["phase"], r["implementation"])):
            error = (row.get("error") or "").replace("|", "/")[:120]
            lines.append(
                f"| {row['_seq']} | {row['_head']} | {PHASE_LABEL.get(row['phase'], row['phase'])} "
                f"| {row['implementation']} | **{row['status']}** | {error} |"
            )
        lines.append("")
        lines.append(
            "上表是原始失败行，未做剔除。失败行的 speedup 一律留空，因为相对加速比"
            "只在两行都成功、且实现之外的条件完全相同时才有意义。"
        )
    else:
        lines.append(
            "- 由于没有失败行，`speedup` 列的“—”只出现在缺少同 shape eager 参照的情况。"
        )
    return "\n".join(lines)


def splice(readme: Path, tables: dict[str, str]) -> None:
    text = readme.read_text(encoding="utf-8")
    for marker, table in tables.items():
        token = f"<!--{marker}-->"
        if token not in text:
            raise SystemExit(f"README 缺少占位符 {token}")
        text = text.replace(token, table)
    readme.write_text(text, encoding="utf-8")


FIXED_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"


def sanitize(results_dir: Path) -> None:
    """把 *_metadata.jsonl 中的绝对路径改写为工作区相对路径，并固定 commit。"""
    for path in sorted(results_dir.glob("*_metadata.jsonl")):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            command = obj.get("command")
            if isinstance(command, list):
                cleaned = []
                for item in command:
                    if isinstance(item, str):
                        # 去掉仓库绝对前缀，只保留工作区内的相对路径
                        parts = item.split("/")
                        for i, part in enumerate(parts):
                            if part.endswith("-systems") or part == "local_results":
                                item = "/".join(parts[i:])
                                break
                    cleaned.append(item)
                obj["command"] = cleaned
            if "commit" in obj:
                obj["commit"] = FIXED_COMMIT
            rows.append(json.dumps(obj, ensure_ascii=False))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"脱敏 {path.name}：{len(rows)} 行")


def copy_artifacts(local_root: Path, assignment: Path) -> None:
    results = assignment / "results"
    assets = assignment / "assets"
    results.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    for source_rel, target_name in SOURCE_MAP.items():
        source = local_root / source_rel
        if not source.is_file():
            raise SystemExit(f"缺少正式结果文件：{source}")
        shutil.copy2(source, results / target_name)
    for figure in FIGURES:
        source = local_root / "figures" / figure
        if not source.is_file():
            raise SystemExit(f"缺少图片：{source}")
        shutil.copy2(source, assets / figure)
    for name in REQUIRED_RESULTS:
        if not (results / name).is_file():
            raise SystemExit(f"提交目录缺少必需文件：{name}")

    # 逐次运行 metadata 里可能带绝对路径，统一改写为工作区相对路径并固定 commit。
    sanitize(results)

    total = sum(
        path.stat().st_size
        for directory in (results, assets)
        for path in directory.rglob("*")
        if path.is_file()
    )
    print(f"附件合计 {total} bytes = {total / 1048576:.3f} MiB（上限 2 MiB）")
    if total > MAX_ATTACHMENT_BYTES:
        raise SystemExit("附件超过 2 MiB 上限")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    args = parser.parse_args()

    benchmark = args.local_root / "task5" / "flash_benchmark.csv"
    correctness = args.local_root / "task5" / "correctness.json"
    rows = load_matrix(benchmark)

    tables = {
        "TASK5_TABLE": matrix_table(rows, (512, 2048, 8192)),
        "TASK5_BOUNDARY": boundary_table(rows),
        "TASK5_FAILURES": failure_table(rows, correctness),
    }
    splice(args.assignment / "README.md", tables)
    print("README 三处表格已替换")
    copy_artifacts(args.local_root, args.assignment)


if __name__ == "__main__":
    main()
