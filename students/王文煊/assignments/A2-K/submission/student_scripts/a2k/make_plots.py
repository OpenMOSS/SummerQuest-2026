"""A2-K: generate compressed PNG plots from local_results CSVs.

Produces (in the given output assets dir):
  1. checkpointing_tradeoff.png  - peak memory vs step time by block size
  2. flash_speedup.png           - p50 speedup of compiled/Triton vs eager
  3. attention_latency_scaling.png - eager attention p50 vs seq length

Run:
    .venv/bin/python student_scripts/a2k/make_plots.py <assets_dir>
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOCAL = os.path.join("local_results", "a2k")


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_checkpointing(out):
    rows = [r for r in read_csv(os.path.join(LOCAL, "checkpointing.csv")) if r["status"] == "ok" and r["context_length"] == "1024"]
    rows.sort(key=lambda r: int(r["checkpoint_block_size"]) if r["checkpoint_block_size"] != "none" else 0)
    labels = ["none" if r["checkpoint_block_size"] == "none" else f"bs={r['checkpoint_block_size']}" for r in rows]
    mem = [float(r["peak_allocated_mib"]) for r in rows]
    lat = [float(r["step_time_ms_p50"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(labels, mem, color="#4C72B0", alpha=0.85)
    ax1.set_ylabel("peak allocated (MiB)", color="#4C72B0")
    ax1.set_xlabel("checkpoint block size (ctx=1024, medium)")
    ax2 = ax1.twinx()
    ax2.plot(labels, lat, "o-", color="#C44E52", lw=2)
    ax2.set_ylabel("step time p50 (ms)", color="#C44E52")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_speedup(out):
    rows = [r for r in read_csv(os.path.join(LOCAL, "flash_benchmark.csv")) if r["status"] == "ok" and r["implementation"] != "eager" and r["speedup_vs_eager"]]
    core = [r for r in rows if r["phase"] == "fwd_bwd"]
    labels = [f"{r['implementation']}\n{r['seq_len']}x{r['head_dim']}" for r in core]
    vals = [float(r["speedup_vs_eager"]) for r in core]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#55A868" if r["implementation"] == "triton" else "#8172B3" for r in core]
    ax.bar(labels, vals, color=colors)
    ax.axhline(1.0, color="k", lw=1, ls="--")
    ax.set_ylabel("speedup vs eager (p50, fwd+bwd)")
    ax.set_title("compiled / Triton FlashAttention-2 vs eager explicit attention")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_latency_scaling(out):
    rows = [r for r in read_csv(os.path.join(LOCAL, "attention_baseline.csv")) if r["status"] == "ok" and r["phase"] == "fwd_bwd"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in ["64", "128"]:
        sub = sorted([r for r in rows if r["head_dim"] == d], key=lambda r: int(r["seq_len"]))
        ax.plot([int(r["seq_len"]) for r in sub], [float(r["p50_ms"]) for r in sub], "o-", label=f"d={d}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("p50 fwd+bwd latency (ms)")
    ax.set_title("explicit PyTorch attention (causal, bf16)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    plot_checkpointing(os.path.join(out_dir, "checkpointing_tradeoff.png"))
    plot_speedup(os.path.join(out_dir, "flash_speedup.png"))
    plot_latency_scaling(os.path.join(out_dir, "attention_latency_scaling.png"))
    for f in sorted(os.listdir(out_dir)):
        print(f, os.path.getsize(os.path.join(out_dir, f)) // 1024, "KB")


if __name__ == "__main__":
    main()
