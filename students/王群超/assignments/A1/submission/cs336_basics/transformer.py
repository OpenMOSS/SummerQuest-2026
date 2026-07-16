from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .nn_utils import (
    Linear,
    Embedding,
    RMSNorm,
    SwiGLU,
    RotaryPositionalEmbedding,
    scaled_dot_product_attention,
)


def _make_causal_mask(T: int, device: torch.device) -> Tensor:
    """因果掩码 shape (T, T)，True=允许注意力（下三角含主对角线）。"""
    return torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(..., T, d_model) -> (... , num_heads, T, d_k)，d_k = d_model / num_heads。"""
    d_model = x.shape[-1]
    d_k = d_model // num_heads
    return x.unflatten(-1, (num_heads, d_k)).transpose(-3, -2)


def _merge_heads(x: Tensor) -> Tensor:
    """(..., num_heads, T, d_k) -> (... , T, d_model)。"""
    x = x.transpose(-3, -2)
    return x.flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """因果多头自注意力（可选 RoPE）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(
                self.d_k, rope_theta, max_seq_len, device=device
            )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """x: (... , T, d_model)，返回 (... , T, d_model)。"""
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = _split_heads(Q, self.num_heads)
        K = _split_heads(K, self.num_heads)
        V = _split_heads(V, self.num_heads)

        if self.rope is not None:
            T = Q.size(-2)
            if token_positions is None:
                token_positions = torch.arange(
                    T, device=x.device, dtype=torch.long
                ).view(1, T)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 因果掩码：shape (T, T)
        T = Q.size(-2)
        mask = _make_causal_mask(T, x.device)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = _merge_heads(attn_out)
        return self.output_proj(attn_out)


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer 块：RMSNorm→MHA(+RoPE)→残差；RMSNorm→SwiGLU→残差。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """完整 Transformer 语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, context_length, rope_theta,
                    device=device, dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        """in_indices: (batch_size, seq_len)，返回 (batch_size, seq_len, vocab_size)。"""
        x = self.token_embeddings(in_indices)
        T = x.size(-2)
        pos = torch.arange(T, device=x.device, dtype=torch.long).view(1, T)

        for layer in self.layers:
            x = layer(x, pos)

        x = self.ln_final(x)
        return self.lm_head(x)


# ---- 兼容 adapters.py 的纯函数接口（接受显式权重字典） ----

def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """无 RoPE 的多头自注意力纯函数。"""
    layer = MultiHeadSelfAttention(
        d_model, num_heads, max_seq_len=in_features.size(-2), rope_theta=None,
        device=q_proj_weight.device, dtype=q_proj_weight.dtype,
    )
    layer.q_proj.weight.data = q_proj_weight
    layer.k_proj.weight.data = k_proj_weight
    layer.v_proj.weight.data = v_proj_weight
    layer.output_proj.weight.data = o_proj_weight
    return layer(in_features)


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    """带 RoPE 的多头自注意力纯函数。"""
    layer = MultiHeadSelfAttention(
        d_model, num_heads, max_seq_len, rope_theta=theta,
        device=q_proj_weight.device, dtype=q_proj_weight.dtype,
    )
    layer.q_proj.weight.data = q_proj_weight
    layer.k_proj.weight.data = k_proj_weight
    layer.v_proj.weight.data = v_proj_weight
    layer.output_proj.weight.data = o_proj_weight
    return layer(in_features, token_positions)


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict,
    in_features: Tensor,
) -> Tensor:
    """Transformer 块纯函数。"""
    layer = TransformerBlock(
        d_model, num_heads, d_ff, max_seq_len, theta,
        device=in_features.device, dtype=in_features.dtype,
    )
    state = {
        "ln1.weight": weights["ln1.weight"],
        "attn.q_proj.weight": weights["attn.q_proj.weight"],
        "attn.k_proj.weight": weights["attn.k_proj.weight"],
        "attn.v_proj.weight": weights["attn.v_proj.weight"],
        "attn.output_proj.weight": weights["attn.output_proj.weight"],
        "ln2.weight": weights["ln2.weight"],
        "ffn.w1.weight": weights["ffn.w1.weight"],
        "ffn.w2.weight": weights["ffn.w2.weight"],
        "ffn.w3.weight": weights["ffn.w3.weight"],
    }
    layer.load_state_dict(state)
    return layer(in_features)


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict,
    in_indices: Tensor,
) -> Tensor:
    """完整 LM 纯函数。"""
    
    weight_dtype = weights["token_embeddings.weight"].dtype
    layer = TransformerLM(
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
        device=in_indices.device, dtype=weight_dtype,
    )
    layer.load_state_dict(weights)
    return layer(in_indices)
