"""Activation checkpointing 的可复用模型前向实现。"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


def forward_with_checkpoint(model: torch.nn.Module, tokens: torch.Tensor, block_size: int | None) -> torch.Tensor:
    """按连续层区间执行 Transformer 前向。"""
    if block_size is not None and block_size <= 0:
        raise ValueError("block_size 必须是正整数或 None")

    hidden = model.token_embeddings(tokens)
    if block_size is None:
        for layer in model.layers:
            hidden = layer(hidden)
    else:
        # 只保存区间输入，反向传播时重算区间内部的 activation。
        for start in range(0, len(model.layers), block_size):
            end = min(start + block_size, len(model.layers))

            def run_block(value: torch.Tensor, start: int = start, end: int = end) -> torch.Tensor:
                for layer_index in range(start, end):
                    value = model.layers[layer_index](value)
                return value

            hidden = checkpoint(run_block, hidden, use_reentrant=False)

    return model.lm_head(model.ln_final(hidden))
