from __future__ import annotations

import math


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """返回带线性 warmup 与 cosine annealing 的第 ``it`` 步学习率。"""

    if it < 0:
        raise ValueError("it must not be negative.")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must not be negative.")
    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters.")
    if min_learning_rate < 0.0 or max_learning_rate < 0.0:
        raise ValueError("learning rates must not be negative.")
    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate must not exceed max_learning_rate.")

    # Warmup 从 0 开始，在 it == warmup_iters 时恰好到达 max_learning_rate。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # Cosine 区间包含两个端点：起点为 max_learning_rate，终点为 min_learning_rate。
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine_scale * (max_learning_rate - min_learning_rate)

    # Annealing 完成后保持最终学习率，避免出现周期性重启。
    return min_learning_rate
