"""顺序启动 TinyStories 的学习率实验。

默认只打印计划，传入 --execute 才会真正占用 GPU。通过命令行参数指定
GPU 数和每卡 batch；每个 run 保持相同的全局 batch 与 token 预算，因而曲线
差异可以归因于 peak learning rate。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_with_host_nvidia.sh"


@dataclass(frozen=True)
class SweepRun:
    """一条学习率实验的不可变描述。"""

    label: str
    max_lr: float
    steps: int
    warmup_steps: int
    constant_after_warmup: bool = False

    @property
    def min_lr(self) -> float:
        """完整收敛实验保持衰减比例；边界实验则固定 LR。"""
        return self.max_lr if self.constant_after_warmup else self.max_lr / 10


# 第一阶段：以较小、相同的 token 预算寻找稳定性边界。warmup 后固定学习率，
# 而非余弦衰减；否则高 LR 只短暂出现，无法可靠判断它是否处于发散边缘。
# 筛选完成后，应将最佳的稳定 LR 用更长训练预算和正式的 cosine schedule 重新训练。
FULL_RUNS = (
    SweepRun("screen_lr3e-4", 3e-4, 1000, 50, constant_after_warmup=True),
    SweepRun("screen_lr6e-4", 6e-4, 1000, 50, constant_after_warmup=True),
    SweepRun("screen_lr1e-3", 1e-3, 1000, 50, constant_after_warmup=True),
    SweepRun("screen_lr3e-3", 3e-3, 1000, 50, constant_after_warmup=True),
    SweepRun("screen_lr1e-2", 1e-2, 1000, 50, constant_after_warmup=True),
)

# 第二阶段：在 1e-2 仍能缓慢下降后，继续按对数尺度提高 LR，确保曲线包含
# 至少一个真正发散的 run。两组都保持 warmup 后固定 LR。
EDGE_RUNS = (
    SweepRun("edge_lr3e-2", 3e-2, 1000, 50, constant_after_warmup=True),
    SweepRun("edge_lr1e-1", 1e-1, 1000, 50, constant_after_warmup=True),
)


def command_for(run: SweepRun, nproc_per_node: int, batch_size: int) -> list[str]:
    """构造一个不依赖 shell 展开的、可复现的 DDP 启动命令。"""
    return [
        str(LAUNCHER),
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "scripts/train_lm.py",
        "--config-name",
        "TS_lr_sweep",
        f"run_name=lr_sweep/{run.label}",
        f"optimizer.max_lr={run.max_lr:g}",
        f"optimizer.min_lr={run.min_lr:g}",
        f"scheduler.warmup_steps={run.warmup_steps}",
        f"scheduler.cosine_cycle_steps={run.steps}",
        f"training.batch_size={batch_size}",
        f"training.steps={run.steps}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("full", "edge", "all"), default="full")
    parser.add_argument("--nproc-per-node", type=int, default=2, help="DDP 使用的本机 GPU 数，默认 2")
    parser.add_argument("--batch-size", type=int, default=64, help="每张 GPU 的 batch size，默认 64")
    parser.add_argument("--execute", action="store_true", help="实际顺序执行；默认仅打印计划")
    args = parser.parse_args()
    if args.nproc_per_node <= 0 or args.batch_size <= 0:
        raise ValueError("--nproc-per-node and --batch-size must be positive")

    runs = ()
    if args.phase in {"full", "all"}:
        runs += FULL_RUNS
    if args.phase in {"edge", "all"}:
        runs += EDGE_RUNS

    global_batch_size = args.nproc_per_node * args.batch_size
    print(
        f"将运行 {len(runs)} 个实验；每个实验使用 {args.nproc_per_node} 张 GPU，"
        f"每卡 batch={args.batch_size}（global batch={global_batch_size}）。"
    )
    for index, run in enumerate(runs, start=1):
        command = command_for(run, args.nproc_per_node, args.batch_size)
        tokens = run.steps * global_batch_size * 256
        print(
            f"[{index}/{len(runs)}] {run.label}: peak_lr={run.max_lr:g}, "
            f"steps={run.steps}, processed_tokens={tokens:,}"
        )
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)

    if not args.execute:
        print("\n这是 dry run；确认计划后加 --execute 才会开始训练。")


if __name__ == "__main__":
    main()
