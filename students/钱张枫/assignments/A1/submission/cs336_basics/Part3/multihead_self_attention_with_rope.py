from __future__ import annotations

import torch
from einops import rearrange

from cs336_basics.Part3.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.Part3.rotary_positional_embedding import RotaryPositionalEmbedding
from cs336_basics.Part3.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """对 query 和 key 使用 RoPE 的因果多头自注意力层。"""

    max_seq_len: int
    rope: RotaryPositionalEmbedding

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(d_model, num_heads, device=device, dtype=dtype)

        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero.")

        self.max_seq_len = max_seq_len
        # RoPE 作用在每一个 head 的 d_k 维度，而不是合并前的完整 d_model 维度。
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """对 ``(..., sequence_length, d_model)`` 输入执行带 RoPE 的因果自注意力。"""

        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.shape[-1]}.")

        sequence_length = x.shape[-2]
        if token_positions is None:
            # 未指定位置时，默认按当前输入序列的自然位置 [0, ..., sequence_length - 1] 编号。
            token_positions = torch.arange(sequence_length, device=x.device)
        else:
            token_positions = token_positions.to(device=x.device)

        # 将 Q/K/V 拆为多个 head；只有 query 和 key 包含应被旋转的位置信息。
        query = rearrange(self.q_proj(x), "... sequence (heads d_k) -> ... heads sequence d_k", heads=self.num_heads)
        key = rearrange(self.k_proj(x), "... sequence (heads d_k) -> ... heads sequence d_k", heads=self.num_heads)
        value = rearrange(self.v_proj(x), "... sequence (heads d_k) -> ... heads sequence d_k", heads=self.num_heads)
        query = self.rope(query, token_positions)
        key = self.rope(key, token_positions)

        # 与普通自注意力相同，因果 mask 禁止每个位置读取未来 token。
        causal_mask = torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=x.device).tril()
        attended = scaled_dot_product_attention(query, key, value, causal_mask)

        # 合并各 head 后投影回 d_model，输出的 batch-like 与 sequence 维度保持不变。
        merged_heads = rearrange(attended, "... heads sequence d_k -> ... sequence (heads d_k)")
        return self.output_proj(merged_heads)
