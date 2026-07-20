from __future__ import annotations

from typing import Iterable, Optional

import math
import torch
import torch.nn as nn
from torch import Tensor


class Linear(nn.Module):
    """无偏置线性层：y = x @ W^T，权重形状 (d_out, d_in)。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    """嵌入层：权重形状 (num_embeddings, embedding_dim)。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm：x * g / sqrt(mean(x^2) + eps)。"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)
        ms = x32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(ms + self.eps)
        normed = x32 * rms
        return (normed * self.weight).to(orig_dtype)


def silu(in_features: Tensor) -> Tensor:
    """SiLU(x) = x * sigmoid(x)，逐元素。"""
    return in_features * torch.sigmoid(in_features)


class SwiGLU(nn.Module):
    """SwiGLU FFN：W2(SiLU(W1 x) ⊙ W3 x)。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        a = self.w1(x)
        b = self.w3(x)
        c = silu(a) * b
        return self.w2(c)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE：成对旋转每个位置的 Q/K；不旋 V。

    预计算 cos/sin 缓冲区，persistent=False 避免保存到 state_dict。
    """

    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        half = d_k // 2
        # 频率：theta_i = 1 / theta^(2i / d_k), i = 0 .. half-1
        idx = torch.arange(half, dtype=torch.float32, device=device)
        freqs = 1.0 / (theta ** (2.0 * idx / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # (max_seq_len, half)
        angles = torch.outer(positions, freqs)
        self.register_buffer(
            "cos_buf", angles.cos().to(torch.float32), persistent=False
        )
        self.register_buffer(
            "sin_buf", angles.sin().to(torch.float32), persistent=False
        )

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """x 形状 (... seq_len d_k)，token_positions 形状 (... seq_len)。"""
        orig_dtype = x.dtype
        x32 = x.to(torch.float32)

        # 取出当前序列长度对应的 cos/sin，按 token_positions 索引
        # cos_buf / sin_buf: (max_seq_len, half)
        # token_positions: (... seq_len) -> 展平后索引 -> (num_pos, half)
        pos_flat = token_positions.reshape(-1)
        cos = self.cos_buf[pos_flat]  # (num_pos, half)
        sin = self.sin_buf[pos_flat]

        # 恢复到 (... seq_len, half)
        cos = cos.reshape(*token_positions.shape, -1)
        sin = sin.reshape(*token_positions.shape, -1)

        # 在头部补 1，让最后两维 (seq_len, half) 与 x 对齐
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        # repeat_interleave 把 (..., half) 扩展为 (..., d_k)，适配交错配对
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        x_even = x32[..., 0::2]
        x_odd = x32[..., 1::2]
        # 旋转：[-x_odd, x_even]
        rotated = torch.stack([-x_odd, x_even], dim=-1).flatten(-2)

        out = x32 * cos + rotated * sin
        return out.to(orig_dtype)


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """稳定 softmax：减该维最大值，再 exp / sum。"""
    x_max = in_features.max(dim=dim, keepdim=True).values
    shifted = in_features - x_max
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """SDPA：QK^T/sqrt(d_k) → mask(False→-inf) → softmax → @ V。"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask.logical_not(), neg_inf)
    attn = softmax(scores, dim=-1)
    return attn @ V


def cross_entropy(
    inputs: Tensor,
    targets: Tensor,
) -> Tensor:
    """平均交叉熵 loss：-log(softmax(logits)[y]) 的均值。
    inputs: (..., vocab_size)，未归一化 logits。
    targets: (...)，正确类别索引。
    """
    log_probs = torch.log_softmax(inputs, dim=-1)
    nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()


def gradient_clipping(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> None:
    """全局梯度裁剪：所有参数梯度合起来算一个总 L2 norm，超过 max_l2_norm 就等比缩小。"""
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm = torch.stack([p.grad.detach().norm(2) for p in params_with_grad]).norm(2)
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.data.mul_(clip_coef)



def linear(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """纯函数 Linear：weights 形状 (d_out, d_in)。"""
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.weight.data = weights
    return layer(in_features)


def embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    """纯函数 Embedding：weights 形状 (vocab_size, d_model)。"""
    layer = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    layer.weight.data = weights
    return layer(token_ids)


def rmsnorm(
    d_model: int,
    eps: float,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    """纯函数 RMSNorm：weights 形状 (d_model,)。"""
    layer = RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    layer.weight.data = weights
    return layer(in_features)


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    """纯函数 SwiGLU：三权重形状与 Linear 相同。"""
    layer = SwiGLU(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    layer.w1.weight.data = w1_weight
    layer.w2.weight.data = w2_weight
    layer.w3.weight.data = w3_weight
    return layer(in_features)


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    """纯函数 RoPE。"""
    layer = RotaryPositionalEmbedding(
        d_k, theta, max_seq_len, device=in_query_or_key.device
    )
    return layer(in_query_or_key, token_positions)
