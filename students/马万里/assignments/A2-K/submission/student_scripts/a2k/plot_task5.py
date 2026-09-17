"""绘制 A2-K 结果图（任务五性能矩阵 + 任务一 checkpointing 显存权衡）。

产出 PNG 供 Markdown 报告引用（README 要求至少两张图被引用）。图只画**成功**
的运行行；失败的配置通过单独标注体现，不会被静默丢弃。

用法::

    python student_scripts/a2k/plot_task5.py --output-dir local_results/a2k/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASES = ("forward", "backward", "forward_backward")
IMPLEMENTATIONS = ("eager", "compiled", "triton")


def has_cjk_font() -> bool:
    """系统是否有可渲染汉字的字体。

    容器与计算节点经常不装 CJK 字体，缺字体时汉字会变成方块。因此这里探测一次，
    有就用中文标签，没有就整套退回英文——保证图在任何机器上都可读。
    """
    from matplotlib import font_manager

    for font in font_manager.fontManager.ttflist:
        name = font.name.lower()
        if any(key in name for key in ("cjk", "wenquanyi", "wqy", "source han", "heiti",
                                       "simhei", "simsun", "yahei", "noto sans sc")):
            return True
    return False


CJK = has_cjk_font()

TEXT = {
    "zh": {
        "latency_title": "A2-K 任务五：三种实现的 p50 延迟（BF16, batch=1, causal）",
        "speedup_title": "A2-K 任务五：相对 eager 的加速比（缺失的条形代表无等价成功参照）",
        "speedup_ylabel": "speedup vs eager（同 shape/dtype/causal）",
        "memory_title": "A2-K 任务五：峰值显存（eager vs Triton）",
        "memory_ylabel": "peak allocated (MiB)",
        "allocator_line": "allocator 上限 23552 MiB",
        "ckpt_title": "A2-K 任务一：checkpointing 的显存—时间权衡（标签为 block size / none）",
        "ckpt_ylabel": "step p50 (ms)",
        "phase": {"forward": "前向", "backward": "反向", "forward_backward": "前向+反向"},
    },
    "en": {
        "latency_title": "A2-K Task 5: p50 latency of three implementations (BF16, batch=1, causal)",
        "speedup_title": "A2-K Task 5: speedup vs eager (missing bars = no equivalent successful reference)",
        "speedup_ylabel": "speedup vs eager (same shape/dtype/causal)",
        "memory_title": "A2-K Task 5: peak memory (eager vs Triton)",
        "memory_ylabel": "peak allocated (MiB)",
        "allocator_line": "allocator limit 23552 MiB",
        "ckpt_title": "A2-K Task 1: checkpointing memory-time tradeoff (labels = block size / none)",
        "ckpt_ylabel": "step p50 (ms)",
        "phase": {"forward": "forward", "backward": "backward", "forward_backward": "forward+backward"},
    },
}
L = TEXT["zh" if CJK else "en"]


def phase_label(phase: str) -> str:
    return L["phase"].get(phase, phase)


def to_float(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_latency(rows: list[dict], output_dir: Path) -> Path | None:
    """按 (head_dim, phase) 分面，画各实现随序列长度的 p50 延迟。"""
    usable = [r for r in rows if r.get("status") == "ok" and to_float(r.get("p50_ms")) is not None]
    if not usable:
        return None
    head_dims = sorted({int(r["head_dim"]) for r in usable})
    fig, axes = plt.subplots(len(head_dims), len(PHASES), figsize=(5 * len(PHASES), 3.2 * len(head_dims)),
                             squeeze=False)
    for row_index, head_dim in enumerate(head_dims):
        for col_index, phase in enumerate(PHASES):
            ax = axes[row_index][col_index]
            for implementation in IMPLEMENTATIONS:
                series = sorted(
                    ((int(r["sequence_length"]), to_float(r["p50_ms"])) for r in usable
                     if int(r["head_dim"]) == head_dim and r["phase"] == phase
                     and r["implementation"] == implementation),
                    key=lambda item: item[0],
                )
                if series:
                    ax.plot([x for x, _ in series], [y for _, y in series], marker="o", label=implementation)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"head_dim={head_dim} · {phase_label(phase)}")
            ax.set_xlabel("sequence length")
            ax.set_ylabel("p50 latency (ms)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(L["latency_title"], y=1.0)
    fig.tight_layout()
    target = output_dir / "task5_latency.png"
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_speedup(rows: list[dict], output_dir: Path) -> Path | None:
    usable = [r for r in rows if r.get("status") == "ok" and to_float(r.get("speedup_vs_eager")) is not None]
    if not usable:
        return None
    labels = sorted({(int(r["sequence_length"]), int(r["head_dim"]), r["phase"]) for r in usable})
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    positions = range(len(labels))
    for offset, implementation in ((-width / 2, "triton"), (width / 2, "compiled")):
        values = []
        for sequence_length, head_dim, phase in labels:
            found = next((to_float(r["speedup_vs_eager"]) for r in usable
                          if r["implementation"] == implementation
                          and int(r["sequence_length"]) == sequence_length
                          and int(r["head_dim"]) == head_dim and r["phase"] == phase), None)
            values.append(found if found is not None else 0.0)
        ax.bar([p + offset for p in positions], values, width=width, label=f"{implementation} / eager")
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{s}\nd={h}\n{phase_label(p)}" for s, h, p in labels], fontsize=7)
    ax.set_ylabel(L["speedup_ylabel"])
    ax.set_title(L["speedup_title"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    target = output_dir / "task5_speedup.png"
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_memory(rows: list[dict], output_dir: Path) -> Path | None:
    usable = [r for r in rows if r.get("status") == "ok" and to_float(r.get("peak_allocated_mib")) is not None]
    if not usable:
        return None
    labels = sorted({(int(r["sequence_length"]), int(r["head_dim"]), r["phase"]) for r in usable})
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels)), 5))
    positions = range(len(labels))
    for offset, implementation in ((-width / 2, "eager"), (width / 2, "triton")):
        values = []
        for sequence_length, head_dim, phase in labels:
            found = next((to_float(r["peak_allocated_mib"]) for r in usable
                          if r["implementation"] == implementation
                          and int(r["sequence_length"]) == sequence_length
                          and int(r["head_dim"]) == head_dim and r["phase"] == phase), None)
            values.append(found if found is not None else 0.0)
        ax.bar([p + offset for p in positions], values, width=width, label=implementation)
    ax.axhline(23552, color="red", linewidth=1, linestyle="--", label=L["allocator_line"])
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{s}\nd={h}\n{phase_label(p)}" for s, h, p in labels], fontsize=7)
    ax.set_ylabel(L["memory_ylabel"])
    ax.set_title(L["memory_title"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    target = output_dir / "task5_memory.png"
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_checkpointing(path: Path, output_dir: Path) -> Path | None:
    """任务一：显存与时间的权衡曲线（block size 扫描）。"""
    rows = [r for r in load_rows(path) if r.get("status") == "ok"
            and to_float(r.get("peak_allocated_mib")) is not None
            and to_float(r.get("step_time_ms_p50")) is not None]
    if not rows:
        return None
    contexts = sorted({int(r["context_length"]) for r in rows})
    fig, axes = plt.subplots(1, len(contexts), figsize=(6 * len(contexts), 4.2), squeeze=False)
    for index, context in enumerate(contexts):
        ax = axes[0][index]
        series = sorted(
            ((r.get("checkpoint_block_size") or "none", to_float(r["peak_allocated_mib"]),
              to_float(r["step_time_ms_p50"])) for r in rows if int(r["context_length"]) == context),
            key=lambda item: (item[0] == "none", 1e9 if item[0] == "none" else int(item[0])),
        )
        names = [item[0] for item in series]
        ax.plot([item[1] for item in series], [item[2] for item in series], marker="o")
        for name, allocated, latency in series:
            ax.annotate(name, (allocated, latency), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("peak allocated (MiB)")
        ax.set_ylabel(L["ckpt_ylabel"])
        ax.set_title(f"context length = {context}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(L["ckpt_title"])
    fig.tight_layout()
    target = output_dir / "task1_checkpointing_tradeoff.png"
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k/figures"))
    parser.add_argument("--benchmark", type=Path, default=Path("local_results/a2k/task5/flash_benchmark.csv"))
    parser.add_argument("--checkpointing", type=Path, default=Path("local_results/a2k/checkpointing.csv"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.benchmark)
    produced = [plot_latency(rows, args.output_dir), plot_speedup(rows, args.output_dir),
                plot_memory(rows, args.output_dir), plot_checkpointing(args.checkpointing, args.output_dir)]
    made = [path for path in produced if path is not None]
    if not made:
        print("没有可画的数据（性能矩阵与 checkpointing 结果都为空）")
        return
    for path in made:
        print(f"已生成 {path}  ({path.stat().st_size / 1024:.1f} KiB)")
    if len(made) < 2:
        print("提示: 报告要求至少两张被引用的图，当前不足；请先补跑对应任务的矩阵")


if __name__ == "__main__":
    main()
