# cs336_basics/training.py

import math
from collections.abc import Iterable
import torch
from torch import Tensor
from torch.nn import Parameter
from os import PathLike
from typing import BinaryIO
from torch import nn
from torch.optim import Optimizer
import numpy as np


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute the mean cross-entropy loss.

    Args:
        logits:
            Shape (..., vocab_size).
            最后一维是词表维度。

        targets:
            Shape (...).
            每个位置保存正确 token 的词表下标。

    Returns:
        Scalar tensor containing the mean loss.
    """
    # 每个位置都沿词表维度取最大值。
    # keepdim=True 保留最后一个维度，方便广播。
    max_logits = logits.max(dim=-1, keepdim=True).values

    # 减去最大值，避免 exp(logits) 数值溢出。
    shifted_logits = logits - max_logits

    # log(sum(exp(logits)))，shape 为 logits.shape[:-1]
    log_normalizer = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )

    # targets 原来是 (...).
    # unsqueeze 后是 (..., 1)，才能沿最后一维 gather。
    target_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 每个位置的 loss：
    # -logit_target + log(sum(exp(logits)))
    losses = log_normalizer - target_logits

    # 对 batch、sequence 等所有位置取平均。
    return losses.mean()


def gradient_clipping(
    parameters: Iterable[Parameter],
    max_l2_norm: float,
) -> None:
    """
    Clip all parameter gradients using one global L2 norm.

    This function modifies parameter.grad in place.
    """
    # parameters 经常是 model.parameters()，它是生成器。
    # 因此先把有效梯度收集起来，避免重复遍历生成器。
    grads = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    if len(grads) == 0:
        return

    # 使用 float32 计算平方和，避免 fp16/bf16 下数值不稳定。
    total_squared_norm = torch.zeros(
        (),
        device=grads[0].device,
        dtype=torch.float32,
    )

    for grad in grads:
        total_squared_norm += grad.detach().float().pow(2).sum()

    global_norm = torch.sqrt(total_squared_norm)

    # 只有超过阈值时才缩放。
    if global_norm > max_l2_norm:
        scale = max_l2_norm / (global_norm + 1e-6)

        with torch.no_grad():
            for grad in grads:
                grad.mul_(
                    scale.to(
                        device=grad.device,
                        dtype=grad.dtype,
                    )
                )


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Compute learning rate using linear warmup followed by cosine decay.

    Args:
        it:
            Current training iteration.

        max_learning_rate:
            Peak learning rate reached after warmup.

        min_learning_rate:
            Final minimum learning rate.

        warmup_iters:
            Number of warmup iterations.

        cosine_cycle_iters:
            Iteration at which cosine decay reaches min_learning_rate.
    """
    # 第一段：线性 warmup
    if it < warmup_iters:
        return (
            it / warmup_iters
        ) * max_learning_rate

    # 第二段：cosine decay
    if it <= cosine_cycle_iters:
        progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        cosine_value = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_learning_rate
            + cosine_value
            * (max_learning_rate - min_learning_rate)
        )

    # 第三段：保持最小学习率
    return min_learning_rate

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | PathLike | BinaryIO,
) -> None:
    """
    Save all information needed to resume training.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | PathLike | BinaryIO,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """
    Load model and optimizer state, then return the saved iteration.
    """
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    return int(checkpoint["iteration"])

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Randomly sample language-model training examples.

    Args:
        dataset:
            One-dimensional array of token IDs.

        batch_size:
            Number of independent sequences.

        context_length:
            Number of input tokens per sequence.

        device:
            Device for the returned tensors.

    Returns:
        x:
            Shape (batch_size, context_length).

        y:
            Shape (batch_size, context_length), where each token is the
            next-token target for the corresponding token in x.
    """
    if len(dataset) <= context_length:
        raise ValueError(
            "Dataset must contain at least context_length + 1 tokens"
        )

    # torch.randint 的 high 是不包含的。
    #
    # 最后一个合法起点：
    # len(dataset) - context_length - 1
    #
    # 因此 high 应为：
    # len(dataset) - context_length
    start_indices = torch.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,),
    )

    x_numpy = np.stack(
        [
            dataset[
                start : start + context_length
            ]
            for start in start_indices.tolist()
        ]
    )

    y_numpy = np.stack(
        [
            dataset[
                start + 1 : start + context_length + 1
            ]
            for start in start_indices.tolist()
        ]
    )

    # torch.tensor 会复制数据，因此对普通 ndarray 和 np.memmap 都稳妥。
    x = torch.tensor(
        x_numpy,
        dtype=torch.long,
        device=device,
    )

    y = torch.tensor(
        y_numpy,
        dtype=torch.long,
        device=device,
    )

    return x, y