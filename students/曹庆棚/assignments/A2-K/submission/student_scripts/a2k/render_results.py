from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--flash", type=Path, required=True)
    p.add_argument("--compile", type=Path)
    p.add_argument("--memory", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = rows(args.checkpoint)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for context, color in (("1024", "#2563eb"), ("2048", "#dc2626")):
        data = [r for r in checkpoint if r["context_length"] == context]
        labels = ["none" if r["checkpoint_block_size"] == "0" else r["checkpoint_block_size"] for r in data]
        axes[0].plot(labels, [float(r["peak_allocated_mib"]) / 1024 for r in data], marker="o", color=color, label=f"S={context}")
        axes[1].plot(labels, [float(r["step_time_ms_p50"]) for r in data], marker="o", color=color, label=f"S={context}")
    axes[0].set_ylabel("Peak allocated (GiB)")
    axes[1].set_ylabel("Training-step p50 (ms)")
    for ax in axes:
        ax.set_xlabel("Checkpoint block size")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "checkpoint_tradeoff.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    flash = rows(args.flash)
    data = [r for r in flash if r["phase"] == "forward" and r["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    styles = {"eager": ("#dc2626", "o"), "compiled": ("#2563eb", "s"), "triton": ("#059669", "^")}
    for i, dim in enumerate(("64", "128")):
        for impl in ("eager", "compiled", "triton"):
            subset = sorted((r for r in data if r["head_dim"] == dim and r["implementation"] == impl), key=lambda r: int(r["seq_len"]))
            if not subset:
                continue
            color, marker = styles[impl]
            axes[i].plot([int(r["seq_len"]) for r in subset], [float(r["p50_ms"]) for r in subset], marker=marker, color=color, label=impl)
        axes[i].set_title(f"head_dim={dim}")
        axes[i].set_xlabel("Sequence length")
        axes[i].set_yscale("log")
        axes[i].grid(alpha=0.25)
        axes[i].legend()
    axes[0].set_ylabel("Forward p50 (ms, log scale)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "flash_forward_latency.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    if args.compile is not None:
        compile_rows = rows(args.compile)
        shapes = [("512", "64"), ("2048", "128"), ("8192", "128")]
        labels = [f"S={s}\nD={d}" for s, d in shapes]
        eager = []
        compiled = []
        cold = []
        steady = []
        for seq, dim in shapes:
            e = next(r for r in compile_rows if r["model"] == "attention" and r["implementation"] == "eager"
                     and r["seq_len"] == seq and r["head_dim"] == dim and r["phase"] == "forward")
            c = next(r for r in compile_rows if r["model"] == "attention" and r["implementation"] == "compiled"
                     and r["seq_len"] == seq and r["head_dim"] == dim and r["phase"] == "forward")
            eager.append(float(e["p50_ms"]))
            compiled.append(float(c["p50_ms"]))
            cold.append(float(c["cold_start_ms"]))
            steady.append(float(c["p50_ms"]))
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
        x = list(range(len(labels)))
        width = 0.36
        axes[0].bar([i - width / 2 for i in x], eager, width, label="eager", color="#dc2626")
        axes[0].bar([i + width / 2 for i in x], compiled, width, label="compiled", color="#2563eb")
        axes[0].set_xticks(x, labels)
        axes[0].set_ylabel("Forward p50 (ms)")
        axes[0].set_title("Steady-state latency")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend()
        axes[1].bar([i - width / 2 for i in x], cold, width, label="cold start", color="#f59e0b")
        axes[1].bar([i + width / 2 for i in x], steady, width, label="steady state", color="#059669")
        axes[1].set_xticks(x, labels)
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Latency (ms, log scale)")
        axes[1].set_title("Compiled attention: compile cost vs reuse")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "compile_cold_steady.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    if args.memory is not None:
        memory = json.loads(args.memory.read_text())
        data = [r for r in flash if r["phase"] == "forward" and r["status"] == "ok"
                and r["head_dim"] == "64" and r["seq_len"] in {"8192", "16384"}
                and r["implementation"] in {"eager", "triton"}]
        labels = ["S=8192\neager", "S=8192\nTriton", "S=16384\neager", "S=16384\nTriton"]
        values = []
        for seq, impl in (("8192", "eager"), ("8192", "triton"), ("16384", "eager"), ("16384", "triton")):
            row = next(r for r in data if r["seq_len"] == seq and r["implementation"] == impl)
            values.append(float(row["peak_allocated_mib"]) / 1024)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        bars = ax.bar(labels, values, color=["#dc2626", "#059669", "#dc2626", "#059669"])
        ax.axhline(memory["allocator"]["allocator_limit_mib"] / 1024, color="#7c3aed", linestyle="--",
                   label="23 GiB allocator limit")
        ax.set_ylabel("Peak allocated (GiB)")
        ax.set_title("Long-sequence forward memory (head_dim=64)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "flash_memory.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
