from __future__ import annotations

import torch
from einops import rearrange
from torch import nn

from cs336_basics.Part3.linear import Linear
from cs336_basics.Part3.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    """不含 RoPE 的因果多头自注意力层。"""

    d_model: int
    num_heads: int
    d_k: int
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    output_proj: Linear

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 每个投影一次性计算所有头；随后在 forward 中再拆分出 head 维度。
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对形状为 ``(..., sequence_length, d_model)`` 的输入执行因果自注意力。"""

        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.shape[-1]}.")

        sequence_length = x.shape[-2]

        # 先做 Q/K/V 投影，再把 d_model 拆成 (num_heads, d_k)，保留全部 batch-like 维度。
        query = rearrange(self.q_proj(x), "... sequence (heads d_k) -> ... heads sequence d_k", heads=self.num_heads)
        key = rearrange(self.k_proj(x), "... sequence (heads d_k) -> ... heads sequence d_k", heads=self.num_heads)
        value = rearrange(self.v_proj(x), "... sequence (heads d_k) -> ... heads sequence d_k", heads=self.num_heads)

        # 第 i 个位置只能关注 [0, i]；二维 mask 会自动广播到 batch 与 head 维度。
        causal_mask = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=x.device).tril()
        attended = scaled_dot_product_attention(query, key, value, causal_mask)

        # 合并各头的输出后执行最终线性投影，恢复原始最后两维的布局。
        merged_heads = rearrange(attended, "... heads sequence d_k -> ... sequence (heads d_k)")
        return self.output_proj(merged_heads)
