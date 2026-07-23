"""顺序运行 TinyStories 的 batch-size sweep。

每个 run 固定 processed-token 预算，并使用 AdamW 的平方根学习率缩放作为
初始值。默认 dry run；传入 --execute 后才会占用 GPU。
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_with_host_nvidia.sh"
CONTEXT_LENGTH = 256
DEFAULT_BATCHES = (1, 16, 64, 128, 256)


def parse_positive_ints(value: str) -> tuple[int, ...]:
    """解析逗号分隔的正整数列表。"""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是逗号分隔的整数，例如 1,8,64") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("batch size 必须全部为正整数")
    return values


def parse_positive_floats(value: str) -> tuple[float, ...]:
    """解析逗号分隔的正浮点数 LR 倍率。"""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是逗号分隔的数，例如 0.5,1,2") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("LR 倍率必须全部为正数")
    return values


def label_number(value: float) -> str:
    """生成可作为 Hydra run_name 一部分的稳定浮点标签。"""
    return f"{value:.2g}".replace(".", "p").replace("+", "")


def command_for(
    *,
    per_gpu_batch: int,
    nproc_per_node: int,
    steps: int,
    warmup_steps: int,
    peak_lr: float,
    min_lr_ratio: float,
    eval_interval: int,
) -> list[str]:
    """为一个 batch/LR 组合构造无 shell 展开的 DDP 启动命令。"""
    run_name = f"batch_sweep/batch_{per_gpu_batch:03d}_lr_{label_number(peak_lr)}"
    return [
        str(LAUNCHER),
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "scripts/train_lm.py",
        "--config-name",
        "TS_batch_sweep",
        f"run_name={run_name}",
        f"optimizer.max_lr={peak_lr:g}",
        f"optimizer.min_lr={peak_lr * min_lr_ratio:g}",
        f"scheduler.warmup_steps={warmup_steps}",
        f"scheduler.cosine_cycle_steps={steps}",
        f"training.batch_size={per_gpu_batch}",
        f"training.steps={steps}",
        f"training.log_interval={eval_interval}",
        f"training.eval_interval={eval_interval}",
        f"training.checkpoint_interval={steps}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, default=4, help="DDP 使用的本机 GPU 数，默认 4")
    parser.add_argument(
        "--batches",
        type=parse_positive_ints,
        default=DEFAULT_BATCHES,
        help="每卡 batch 列表，默认 1,16,64,128,256",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="显存 smoke test 测得的最大安全每卡 batch；会附加到 --batches",
    )
    parser.add_argument("--target-tokens", type=int, default=65_536_000, help="每组固定训练 token 数")
    parser.add_argument("--base-global-batch", type=int, default=256, help="baseline 的 global batch")
    parser.add_argument("--base-lr", type=float, default=1e-3, help="baseline 的 peak LR")
    parser.add_argument("--min-lr-ratio", type=float, default=0.06, help="min_lr / peak_lr")
    parser.add_argument(
        "--lr-multipliers",
        type=parse_positive_floats,
        default=(1.0,),
        help="在平方根缩放初值上的额外倍率，例如 0.5,1,2",
    )
    parser.add_argument("--execute", action="store_true", help="实际顺序执行；默认仅打印计划")
    args = parser.parse_args()
    if args.nproc_per_node <= 0 or args.target_tokens <= 0 or args.base_global_batch <= 0 or args.base_lr <= 0:
        raise ValueError("GPU 数、token 预算、baseline batch 和 baseline LR 必须为正数")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.max_batch_size is not None and args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive")

    batches = set(args.batches)
    if args.max_batch_size is not None:
        batches.add(args.max_batch_size)
    batches = tuple(sorted(batches))
    plans: list[tuple[int, int, int, float, int]] = []
    for batch_size in batches:
        global_batch = args.nproc_per_node * batch_size
        # round 让无法整除的最大 batch 也尽可能接近指定 token 预算。
        steps = max(2, round(args.target_tokens / (global_batch * CONTEXT_LENGTH)))
        warmup_steps = max(1, round(steps * 0.05))
        warmup_steps = min(warmup_steps, steps - 1)
        eval_interval = max(1, steps // 20)  # 每条曲线约 20 个验证点。
        scaled_lr = args.base_lr * math.sqrt(global_batch / args.base_global_batch)
        for multiplier in args.lr_multipliers:
            plans.append((batch_size, global_batch, steps, scaled_lr * multiplier, warmup_steps))

    print(f"将运行 {len(plans)} 个实验；每个实验使用 {args.nproc_per_node} 张 GPU。")
    for index, (batch_size, global_batch, steps, peak_lr, warmup_steps) in enumerate(plans, start=1):
        eval_interval = max(1, steps // 20)
        processed_tokens = steps * global_batch * CONTEXT_LENGTH
        command = command_for(
            per_gpu_batch=batch_size,
            nproc_per_node=args.nproc_per_node,
            steps=steps,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            min_lr_ratio=args.min_lr_ratio,
            eval_interval=eval_interval,
        )
        print(
            f"[{index}/{len(plans)}] per_gpu_batch={batch_size}, global_batch={global_batch}, "
            f"steps={steps}, warmup={warmup_steps}, peak_lr={peak_lr:.3g}, "
            f"processed_tokens={processed_tokens:,}"
        )
        print(" ".join(command))
        if args.execute:
            subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)

    if not args.execute:
        print("\n这是 dry run；确认计划后加 --execute 才会开始训练。")


if __name__ == "__main__":
    main()
