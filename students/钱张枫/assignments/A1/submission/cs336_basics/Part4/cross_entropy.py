from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算任意 batch-like 维度上的平均多类别交叉熵损失。

    ``logits`` 的形状为 ``(..., vocab_size)``，``targets`` 的形状为 ``(...)``，
    最后一维之前的所有维度都会视为独立样本并参与最终平均。
    """

    if logits.ndim == 0:
        raise ValueError("logits must include a vocabulary dimension.")
    if logits.shape[-1] == 0:
        raise ValueError("vocab_size must be greater than zero.")
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match the batch-like dimensions of logits.")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("targets must contain integer class indices.")
    if targets.device != logits.device:
        raise ValueError("logits and targets must be on the same device.")

    vocab_size = logits.shape[-1]
    if torch.any(targets < 0) or torch.any(targets >= vocab_size):
        raise ValueError("targets must be valid vocabulary indices.")

    # 低精度训练时在 float32 中完成 log-sum-exp，避免 bf16/fp16 的指数与归一化误差。
    working_logits = logits.to(dtype=torch.float32) if logits.dtype in (torch.float16, torch.bfloat16) else logits

    # 减去每个样本的最大 logit，使 exp 的输入不大于 0，避免数值上溢。
    shifted_logits = working_logits - working_logits.amax(dim=-1, keepdim=True)
    log_normalizer = torch.log(torch.exp(shifted_logits).sum(dim=-1))

    # -log softmax(target) 化简为 log(sum(exp(shifted_logits))) - shifted_target_logit。
    target_logits = shifted_logits.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()
