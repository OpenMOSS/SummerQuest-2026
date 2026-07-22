"""Transformer language-model components implemented from first principles."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


# 手写线性层：权重保存为 (out_features, in_features)，前向时转置后与输入相乘。
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        # x 的最后一维是 in_features，输出最后一维变为 out_features。
        return x @ self.weight.transpose(-1, -2)


# Embedding 本质是按整数 token id 从可学习矩阵中查表。
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


# RMSNorm 不减均值，只按最后一维的均方根缩放；中间计算转 float32 以提高 BF16 稳定性。
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float.square(), dim=-1, keepdim=True) + self.eps)
        return (x_float / rms * self.weight.float()).to(input_dtype)


class IdentityNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


# SwiGLU: W2(SiLU(W1(x)) * W3(x))，门控分支和数值分支逐元素相乘。
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """Two-matrix SiLU FFN used for the parameter-matched ablation."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        # 预先缓存每个位置、每对通道的旋转角，forward 时只做索引与二维旋转。
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dimensions = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimensions / d_k)
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        # 将相邻通道看作二维向量：(even, odd) -> 旋转后的二维向量。
        even = x[..., 0::2]
        odd = x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output


RoPE = RotaryPositionalEmbedding


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    # 减去最大值不改变 softmax，但可避免 exp(大正数) 溢出。
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    numerator = torch.exp(shifted)
    return numerator / torch.sum(numerator, dim=dim, keepdim=True)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    # QK^T 先除以 sqrt(d_head)，避免维度变大时注意力分数过于尖锐。
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return softmax(scores, dim=-1) @ v


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = None
        if theta is not None:
            if max_seq_len is None:
                raise ValueError("max_seq_len is required when RoPE is enabled")
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        batch_shape = x.shape[:-2]

        def split_heads(projected: Tensor) -> Tensor:
            # (batch, seq, d_model) -> (batch, heads, seq, d_head)，让每个 head 独立计算注意力。
            projected = projected.reshape(*batch_shape, sequence_length, self.num_heads, self.d_head)
            return projected.transpose(-3, -2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
                token_positions = token_positions.expand(*batch_shape, sequence_length)
            # 补出 head 维以便 position 在所有 head 上广播。
            positions_with_head_axis = token_positions.unsqueeze(-2)
            q = self.rope(q, positions_with_head_axis)
            k = self.rope(k, positions_with_head_axis)

        # 下三角 causal mask 保证位置 t 不能读取未来 token。
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device))
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        # 拼回多个 head，再用输出投影混合各 head 的信息。
        attended = attended.transpose(-3, -2).contiguous().reshape(*batch_shape, sequence_length, self.d_model)
        return self.output_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None,
        device=None,
        dtype=None,
        norm_mode: str = "pre",
        use_rmsnorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if norm_mode not in {"pre", "post"}:
            raise ValueError("norm_mode must be 'pre' or 'post'")
        self.norm_mode = norm_mode
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        self.ln1 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else norm_cls()
        self.ln2 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else norm_cls()

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        return self.ln2(x + self.ffn(x))


TransformerLayer = TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None,
        device=None,
        dtype=None,
        norm_mode: str = "pre",
        use_rmsnorm: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # ModuleList 创建 num_layers 个独立 block；不能在 forward 中反复复用同一个 block。
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                    norm_mode=norm_mode,
                    use_rmsnorm=use_rmsnorm,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        sequence_length = in_indices.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError("input sequence is longer than context_length")
        # 模型接收整数 token id；tokenizer 属于数据预处理，不在网络 forward 中编码字符串。
        token_positions = torch.arange(sequence_length, device=in_indices.device)
        token_positions = token_positions.expand(*in_indices.shape[:-1], sequence_length)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        # 返回未归一化 logits，cross entropy 会在训练时以更稳定的方式完成 softmax。
        return self.lm_head(self.ln_final(x))
