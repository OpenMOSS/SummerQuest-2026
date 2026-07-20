"""CS336 Assignment 1 的数据采样工具脚手架。

这个文件负责训练语言模型时的 batch 构造。核心实现留给你填写。
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """从一维 token ID 数据集中随机采样语言模型训练 batch。

    Shape 约定：
        dataset:        (num_tokens,)
        x:              (batch_size, context_length)
        y:              (batch_size, context_length)

    目标：
        - x 是从 dataset 中随机截取的连续片段。
        - y 是 x 向右偏移 1 个 token 后的标签。
        - 对每一行，都应该满足 y[row] == x[row] 后面紧跟的 token。

    测试要求：
        - 起始位置必须在合法范围内均匀随机采样。
        - 最大起始位置是 len(dataset) - context_length - 1。
        - 返回 torch.Tensor，并放到传入的 device 上。
        - device 无效时应让 torch 自然抛出 RuntimeError / AssertionError。

    实现提示：
        1. 计算合法起始位置数量。
        2. 随机采样 batch_size 个起始位置。
        3. 对每个起始位置取 dataset[start : start + context_length] 作为 x。
        4. 对每个起始位置取 dataset[start + 1 : start + context_length + 1] 作为 y。
        5. 转成 torch.LongTensor 并移动到 device。
    """
    max_start =len(dataset) - context_length
    starts = np.random.randint(0, max_start, batch_size)
    indices = np.arange(context_length)
    x = dataset[starts[:, None] + indices]
    y = dataset[starts[:, None] + 1 + indices]

    x_tensor = torch.as_tensor(x,dtype=torch.long,device=device)
    y_tensor = torch.as_tensor(y,dtype=torch.long,device=device)

    return x_tensor, y_tensor