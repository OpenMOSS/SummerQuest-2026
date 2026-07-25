from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from profiling.common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures from profiling summary tables.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def save_plot(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_benchmark_summary(df: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty:
        return outputs

    plot_df = df[df["dtype"] == "fp32"].copy()
    if not plot_df.empty:
        plot_df["label"] = plot_df["model_size"] + "\n" + plot_df["mode"]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(plot_df["label"], plot_df["mean_ms"], yerr=plot_df["std_ms"], capsize=4)
        ax.set_title("Benchmark Mean Step Time (FP32)")
        ax.set_ylabel("Mean Time (ms)")
        ax.set_xlabel("Model Size / Mode")
        ax.tick_params(axis="x", rotation=45)
        outputs.append(save_plot(fig, output_dir / "benchmark_fp32_mean_ms.png"))

    mixed_df = df[df["dtype"].isin(["fp32", "bf16"])].copy()
    if not mixed_df.empty:
        grouped = (
            mixed_df.groupby(["model_size", "mode", "dtype"], as_index=False)["mean_ms"]
            .mean()
            .pivot(index=["model_size", "mode"], columns="dtype", values="mean_ms")
            .reset_index()
        )
        if {"fp32", "bf16"}.issubset(grouped.columns):
            grouped["bf16_speedup"] = grouped["fp32"] / grouped["bf16"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(grouped["model_size"] + "\n" + grouped["mode"], grouped["bf16_speedup"])
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
            ax.set_title("BF16 Speedup over FP32")
            ax.set_ylabel("FP32 / BF16")
            ax.tick_params(axis="x", rotation=45)
            outputs.append(save_plot(fig, output_dir / "mixed_precision_bf16_speedup.png"))

    return outputs


def plot_memory_summary(df: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty:
        return outputs

    plot_df = df.copy()
    plot_df["allocated_peak_gib"] = plot_df["allocated_peak_bytes"] / (1024**3)
    plot_df["label"] = (
        plot_df["model_size"]
        + "\nctx="
        + plot_df["context_length"].astype(str)
        + "\n"
        + plot_df["mode"]
        + "\n"
        + plot_df["dtype"]
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["label"], plot_df["allocated_peak_gib"])
    ax.set_title("Peak Allocated Memory")
    ax.set_ylabel("Allocated Peak (GiB)")
    ax.tick_params(axis="x", rotation=45)
    outputs.append(save_plot(fig, output_dir / "memory_allocated_peak_gib.png"))
    return outputs


def plot_nsys_summary(df: pd.DataFrame, output_dir: Path) -> list[str]:
    outputs: list[str] = []
    if df.empty:
        return outputs

    plot_df = df.copy()
    plot_df["label"] = plot_df["source_stats_file"].map(lambda x: Path(x).stem.replace("_stats", ""))
    if plot_df["top_kernel_time"].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(plot_df["label"], plot_df["top_kernel_time"])
        ax.set_title("Top Kernel Time per NSYS Stats Export")
        ax.set_ylabel("Time Value (raw nsys units)")
        ax.tick_params(axis="x", rotation=45)
        outputs.append(save_plot(fig, output_dir / "nsys_top_kernel_time.png"))

    if {"matmul_time", "softmax_time"}.issubset(plot_df.columns):
        compare_df = plot_df[["label", "matmul_time", "softmax_time"]].set_index("label")
        fig, ax = plt.subplots(figsize=(12, 6))
        compare_df.plot(kind="bar", ax=ax)
        ax.set_title("NSYS Matmul vs Softmax Time")
        ax.set_ylabel("Time Value (raw nsys units)")
        ax.tick_params(axis="x", rotation=45)
        outputs.append(save_plot(fig, output_dir / "nsys_matmul_vs_softmax.png"))

    return outputs


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    summary_dir = results_root / "summary"
    benchmark_df = load_csv(summary_dir / "benchmark_summary.csv")
    memory_df = load_csv(summary_dir / "memory_summary.csv")
    nsys_df = load_csv(summary_dir / "nsys_kernel_summary.csv")

    generated_files: list[str] = []
    if benchmark_df is not None:
        generated_files.extend(plot_benchmark_summary(benchmark_df, output_dir))
    if memory_df is not None:
        generated_files.extend(plot_memory_summary(memory_df, output_dir))
    if nsys_df is not None:
        generated_files.extend(plot_nsys_summary(nsys_df, output_dir))

    index_path = output_dir / "generated_figures.txt"
    index_path.write_text("\n".join(generated_files) + ("\n" if generated_files else ""), encoding="utf-8")
    print(f"Saved generated figures under {output_dir}")


if __name__ == "__main__":
    main()
