# student_scripts/a2k/generate_plots.py（修复版）
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    df = pd.read_csv("local_results/flash_benchmark.csv")
    
    os.makedirs("assets", exist_ok=True)
    
    # ========== 图 1: p50 Latency vs Sequence Length ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, head_dim in enumerate([64, 128]):
        ax = axes[idx]
        for impl in ["eager", "compiled", "triton"]:
            for phase in ["forward", "backward", "forward_backward"]:
                sub = df[
                    (df["impl"] == impl) & 
                    (df["head_dim"] == head_dim) & 
                    (df["phase"] == phase) &
                    (df["status"] == "success")
                ].sort_values("seq_len")
                
                if len(sub) > 0:
                    ax.plot(
                        sub["seq_len"], 
                        sub["p50_ms"], 
                        marker='o',
                        label=f"{impl}-{phase}",
                        linewidth=1.5,
                        markersize=4
                    )
        
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("p50 Latency (ms)")
        ax.set_title(f"head_dim={head_dim}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("assets/latency_vs_seqlen.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved assets/latency_vs_seqlen.png")
        
    # 修复版 generate_plots.py 中的图 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, head_dim in enumerate([64, 128]):
        ax = axes[idx]
        
        for phase in ["forward", "backward", "forward_backward"]:
            for impl in ["eager", "triton"]:
                sub = df[
                    (df["impl"] == impl) & 
                    (df["head_dim"] == head_dim) & 
                    (df["phase"] == phase) &
                    (df["status"] == "success")
                ].sort_values("seq_len")
                
                if len(sub) > 0:
                    marker = 'o' if impl == "eager" else 's'
                    linestyle = '-' if phase == 'forward' else ('--' if phase == 'backward' else '-.')
                    ax.plot(
                        sub["seq_len"], 
                        sub["peak_allocated_mib"], 
                        marker=marker,
                        linestyle=linestyle,
                        label=f"{impl}-{phase}",
                        linewidth=1.5,
                        markersize=4
                    )
        
        ax.axhline(23552, color='red', linestyle='--', linewidth=2, label="23 GiB limit")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Peak Allocated Memory (MiB)")
        ax.set_title(f"head_dim={head_dim}")
        ax.set_xscale("log", base=2)
        ax.set_ylim(bottom=0)  # 关键：从 0 开始
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("assets/memory_vs_seqlen.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved assets/memory_vs_seqlen.png")

if __name__ == "__main__":
    main()