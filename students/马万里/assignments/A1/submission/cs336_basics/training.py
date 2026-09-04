import os
import io
import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 1D numpy 整数数组 dataset 中随机采样 batch 个训练样本。

    每个样本是一个长度为 context_length 的输入序列，标签是下一个 token。
    起始索引随机选取，保证不会越界。
    """
    # 最大起始索引，使得能取到 context_length 个输入和 1 个目标
    max_start = len(dataset) - context_length - 1
    if max_start < 0:
        raise ValueError("Dataset too short for the given context length")

    # 随机采样 batch_size 个起始位置
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    inputs = []
    labels = []
    for start in starts:
        input_seq = dataset[start : start + context_length]
        label_seq = dataset[start + 1 : start + context_length + 1]
        inputs.append(input_seq)
        labels.append(label_seq)

    inputs = torch.tensor(np.stack(inputs), dtype=torch.long, device=device)
    labels = torch.tensor(np.stack(labels), dtype=torch.long, device=device)

    return inputs, labels

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]