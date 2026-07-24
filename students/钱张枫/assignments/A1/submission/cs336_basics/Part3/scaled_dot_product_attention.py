from __future__ import annotations

import math

import torch
from einops import einsum


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算支持任意 batch-like 维度的 scaled dot-product attention。

    ``query``、``key`` 和 ``value`` 的形状分别为
    ``(..., queries, d_k)``、``(..., keys, d_k)`` 与
    ``(..., keys, d_v)``。可选的 ``mask`` 应能广播到
    ``(..., queries, keys)``，其中 ``True`` 表示该 key 可以被关注。
    """

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same final dimension (d_k).")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length.")
    if query.shape[-1] == 0:
        raise ValueError("d_k must be greater than zero.")

    # 对每个 query 与每个 key 的最后一维做点积，保留所有 batch-like 维度。
    attention_scores = einsum(query, key, "... queries d_k, ... keys d_k -> ... queries keys")
    attention_scores = attention_scores / math.sqrt(query.shape[-1])

    if mask is not None:
        if mask.dtype is not torch.bool:
            raise TypeError("mask must be a boolean tensor.")

        # False 位置被设为 -inf；softmax 后其概率严格为 0，True 位置重新归一化。
        attention_scores = attention_scores.masked_fill(~mask, -torch.inf)

    # softmax 沿 key 维度归一化，因此每个 query 的有效注意力概率之和为 1。
    attention_probabilities = torch.softmax(attention_scores, dim=-1)

    # 用注意力概率对相同 key 位置的 value 做加权求和，得到每个 query 的输出。
    return einsum(attention_probabilities, value, "... queries keys, ... keys d_v -> ... queries d_v")
