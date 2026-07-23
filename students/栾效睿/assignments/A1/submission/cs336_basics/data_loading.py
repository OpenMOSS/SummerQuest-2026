import random
import torch
import numpy as np


def data_loading(
    token_ids: np.ndarray, batch_size: int, context_length: int, device: str | None = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    assert len(token_ids) >= context_length + 1
    if device is None:
        device = "cpu"
    # 所以每个独立样本总共需要 context_length + 1 的长度
    max_start = len(token_ids) - context_length - 1

    # 随机采样 batch_size 个独立的起始索引
    # 这样它们之间是完全解耦、独立的，符合测试用例的统计分布
    start_indices = [random.randint(0, max_start) for _ in range(batch_size)]

    # 构建输入 X 和输出 Y
    x_list = []
    y_list = []

    for start in start_indices:
        # 提取当前通道的完整序列片段
        chunk = token_ids[start : start + context_length + 1]
        x_list.append(chunk[:-1])  # 输入: 0 到 context_length - 1
        y_list.append(chunk[1:])  # 标签: 1 到 context_length (向后偏移1位)

    # 转换为 Tensor
    x = torch.tensor(np.array(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.array(y_list), dtype=torch.long, device=device)

    return x, y
