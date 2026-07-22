import numpy as np
import torch
from typing import Tuple


def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从一维 token 数组或 NumPy memmap 中随机采样 input 和 target。
    只读取当前 batch 所需的数据，避免将完整数据集转换为张量。
    """
    if len(x) <= context_length:
        raise ValueError(
            f"数据长度 {len(x)} 必须大于 context_length {context_length}"
        )

    # 随机生成合法起点。每个窗口需要 context_length + 1 个 token。
    starts = np.random.randint(
        0,
        len(x) - context_length,
        size=batch_size,
    )

    # 仅从 memmap 中读取当前 batch 所需的小窗口。
    # copy() 将只读或磁盘映射数据转换为可写的内存数组。
    windows = np.stack(
        [x[start : start + context_length + 1] for start in starts]
    ).copy()

    # 只转换当前 batch，并构造错开一个 token 的 input 和 target。
    batch = torch.from_numpy(windows).to(dtype=torch.long)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    return inputs.to(device), targets.to(device)