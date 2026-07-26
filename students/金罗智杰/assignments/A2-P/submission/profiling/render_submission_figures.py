from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render lightweight, public A2-P figures from Nsight CSV summaries.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/compute_profile_large_ctx512_train_step_fp32.png"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def nvtx_duration_ms(rows: list[dict[str, str]], name: str) -> float:
    for row in rows:
        if row["Range"].lstrip(":") == name:
            return int(row["Total Time (ns)"]) / 1e6
    raise KeyError(f"missing NVTX range: {name}")


def kernel_label(name: str) -> str:
    if "CUDAFunctor_add" in name:
        return "Vector add"
    if "AUnaryFunctor" in name and "MulFunctor" in name:
        return "Unary mul"
    if "BinaryFunctor" in name and "MulFunctor" in name:
        return "Binary mul"
    if "DivFunctor" in name:
        return "Division"
    if "direct_copy" in name:
        return "Copy"
    if "sm90_xmma_gemm" in name:
        match = re.search(r"_f32_(nt|tn|nn)_", name)
        layout = match.group(1).upper() if match else "mixed"
        return f"SM90 GEMM {layout}"
    return "Other CUDA kernel"


def render(results_dir: Path, output: Path) -> None:
    prefix = results_dir / "nsys" / "large_ctx512_train_step_fp32"
    nvtx_rows = read_csv(prefix.with_name(f"{prefix.name}_nvtx_sum.csv"))
    kernel_rows = read_csv(prefix.with_name(f"{prefix.name}_cuda_gpu_kern_sum.csv"))[:8]

    phase_names = ["Forward", "Backward", "Optimizer"]
    phase_values = [nvtx_duration_ms(nvtx_rows, name.lower()) for name in phase_names]
    attention_names = ["Scores", "Softmax", "Value"]
    attention_values = [
        nvtx_duration_ms(nvtx_rows, "attention/scores"),
        nvtx_duration_ms(nvtx_rows, "attention/softmax"),
        nvtx_duration_ms(nvtx_rows, "attention/value"),
    ]
    kernel_names = [kernel_label(row["Name"]) for row in kernel_rows]
    kernel_values = [int(row["Total Time (ns)"]) / 1e6 for row in kernel_rows]
    kernel_calls = [int(row["Instances"]) for row in kernel_rows]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.6]})
    figure.suptitle(
        "Nsight Systems: Large Transformer, context 512, batch 4, train-step, FP32",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(0.5, 0.92, "One post-warm-up measurement step; durations are NVTX range totals", ha="center", fontsize=10, color="#475569")

    phase_colors = ["#2563eb", "#7c3aed", "#ea580c"]
    axes[0].bar(phase_names, phase_values, color=phase_colors)
    axes[0].set_title("Stage duration")
    axes[0].set_ylabel("Duration (ms)")
    axes[0].tick_params(axis="x", rotation=20)
    for index, value in enumerate(phase_values):
        axes[0].text(index, value + max(phase_values) * 0.025, f"{value:.1f}", ha="center", fontsize=9)

    attention_colors = ["#0f766e", "#dc2626", "#0891b2"]
    axes[1].bar(attention_names, attention_values, color=attention_colors)
    axes[1].set_title("Attention subranges (36 calls each)")
    axes[1].set_ylabel("Cumulative duration (ms)")
    for index, value in enumerate(attention_values):
        axes[1].text(index, value + max(attention_values) * 0.025, f"{value:.2f}", ha="center", fontsize=9)

    positions = list(range(len(kernel_names)))
    axes[2].barh(positions, kernel_values, color="#334155")
    axes[2].set_yticks(positions, kernel_names)
    axes[2].invert_yaxis()
    axes[2].set_title("Top CUDA kernels by cumulative GPU time")
    axes[2].set_xlabel("Cumulative GPU time (ms)")
    for index, (value, calls) in enumerate(zip(kernel_values, kernel_calls, strict=True)):
        axes[2].text(value + max(kernel_values) * 0.015, index, f"{value:.1f} ms · {calls} calls", va="center", fontsize=8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.2)
        axis.grid(axis="y", alpha=0.15)

    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    render(args.results_dir, args.output)
    print(f"saved figure: {args.output}")


if __name__ == "__main__":
    main()
