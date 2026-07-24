from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from student_scripts.a2k.common import ASSETS_DIR, RESULTS_DIR, ensure_dirs


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def checkpointing_figure(results_dir: Path, assets_dir: Path) -> None:
    rows = [r for r in read_csv(results_dir / "checkpointing.csv") if r.get("status") == "ok"]
    if not rows:
        return
    labels = [r["config_id"] for r in rows]
    memory = [float(r["peak_allocated_mib"]) for r in rows]
    latency = [float(r["step_time_ms_p50"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(labels, memory, color="#4C78A8")
    ax1.set_ylabel("Peak allocated MiB")
    ax1.tick_params(axis="x", rotation=30)
    ax2 = ax1.twinx()
    ax2.plot(labels, latency, color="#F58518", marker="o")
    ax2.set_ylabel("Step p50 ms")
    fig.tight_layout()
    fig.savefig(assets_dir / "checkpointing_memory_time.png", dpi=160)
    plt.close(fig)


def flash_figure(results_dir: Path, assets_dir: Path) -> None:
    rows = [
        r
        for r in read_csv(results_dir / "flash_benchmark.csv")
        if r.get("status") == "ok" and r.get("phase") == "forward-backward" and r.get("sequence_length") in {"512", "2048", "8192", "16384"}
    ]
    if not rows:
        return
    labels = [f"{r['implementation']}\nS={r['sequence_length']},D={r['head_dim']}" for r in rows]
    latency = [float(r["p50_ms"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(labels, latency, color="#54A24B")
    ax.set_ylabel("Forward-backward p50 ms")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(assets_dir / "attention_performance_memory.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--assets-dir", type=Path, default=ASSETS_DIR)
    args = parser.parse_args()
    ensure_dirs()
    args.assets_dir.mkdir(exist_ok=True)
    checkpointing_figure(args.results_dir, args.assets_dir)
    flash_figure(args.results_dir, args.assets_dir)


if __name__ == "__main__":
    main()
