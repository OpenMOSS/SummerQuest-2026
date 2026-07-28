from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_checkpointing(results_dir: Path, assets_dir: Path) -> None:
    rows = [r for r in read_csv(results_dir / "checkpointing.csv") if r.get("status") == "ok" and r.get("context_length") == "1024"]
    if not rows:
        return
    labels = ["none" if r["checkpoint_block_size"] == "" else f"bs={r['checkpoint_block_size']}" for r in rows]
    time_ms = [float(r["step_time_ms_p50"]) for r in rows]
    memory = [float(r["peak_allocated_mib"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(labels, time_ms, marker="o", color="#1f77b4", label="p50 step time")
    ax1.set_ylabel("p50 step time (ms)")
    ax1.set_xlabel("checkpoint block size")
    ax2 = ax1.twinx()
    ax2.plot(labels, memory, marker="s", color="#d62728", label="peak allocated")
    ax2.set_ylabel("peak allocated (MiB)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets_dir / "a2k_checkpointing_tradeoff.png", dpi=160)
    plt.close(fig)


def plot_flash(results_dir: Path, assets_dir: Path) -> None:
    rows = [
        r
        for r in read_csv(results_dir / "flash_benchmark.csv")
        if r.get("status") == "ok" and r.get("phase") == "forward_backward" and r.get("head_dim") == "128" and r.get("sequence_length") in {"512", "2048", "8192", "16384"}
    ]
    if not rows:
        return
    implementations = ["eager", "compiled", "triton"]
    seqs = sorted({int(r["sequence_length"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    for impl in implementations:
        xs, ys = [], []
        for seq in seqs:
            match = [r for r in rows if r["implementation"] == impl and int(r["sequence_length"]) == seq]
            if match:
                xs.append(seq)
                ys.append(float(match[0]["p50_ms"]))
        if xs:
            ax.plot(xs, ys, marker="o", label=impl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("p50 forward+backward latency (ms)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    assets_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets_dir / "a2k_flash_attention_latency.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--assets-dir", type=Path, default=Path("local_results/a2k/assets"))
    args = parser.parse_args()
    plot_checkpointing(args.results_dir, args.assets_dir)
    plot_flash(args.results_dir, args.assets_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
