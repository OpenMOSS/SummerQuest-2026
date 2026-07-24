from __future__ import annotations

import math

import torch
from torch import nn

class Linear(nn.Module):
    """不含 bias 的线性变换，权重按 (out_features, in_features) 存储。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        standard_deviation = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=standard_deviation,
            a=-3 * standard_deviation,
            b=3 * standard_deviation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算 y = xWᵀ；输入的最后一维必须为 in_features。"""
        return x @ self.weight.T


class Embedding(nn.Module):
    """将整数 token ID 映射为对应的 embedding 向量。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回 token_ids 中每个 ID 对应的 embedding，输出最后一维为 embedding_dim。"""
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """对最后一维执行 Root Mean Square Layer Normalization。"""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """以 float32 计算均方根归一化，再还原为输入 dtype。"""
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * rms * self.weight.to(torch.float32)).to(input_dtype)


class SwiGLU(nn.Module):
    """位置独立的 SwiGLU 前馈网络：W2(SiLU(W1x) ⊙ W3x)。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            # 将 8/3 * d_model 向上取整到 64 的倍数，便于硬件高效计算。
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入的最后一维独立执行 SwiGLU，保留所有前导维度。"""
        gate = self.w1(x)
        return self.w2((gate * torch.sigmoid(gate)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """不含门控的前馈网络：W2(SiLU(W1x))。"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(x)
        return self.w2(hidden * torch.sigmoid(hidden))


def silu(x: torch.Tensor) -> torch.Tensor:
    """逐元素计算 SiLU(x) = x * sigmoid(x)。"""
    return x * torch.sigmoid(x)


class RotaryPositionalEmbedding(nn.Module):
    """对查询或键向量的相邻维度对应用旋转位置编码（RoPE）。"""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE 的 d_k 必须为偶数")

        dimensions = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimensions / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        # 这些值是固定的，不应被优化器保存或更新。
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """按 token_positions 指定的位置旋转 x 的最后一维。"""
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        # 若 x 含有额外的 head 维度，将频率表在 sequence 维之前扩展，
        # 使每个 head 共享同一组按位置计算的旋转角。
        while cos.ndim < x.ndim - 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """沿指定维度计算数值稳定的 softmax。"""
    shifted_x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算支持任意前导 batch 维度的缩放点积注意力。"""
    d_k = query.shape[-1]
    attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    return attention_weights @ value


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """计算任意 batch-like 前导维度上的平均、数值稳定交叉熵。"""
    shifted_logits = logits - logits.max(dim=-1, keepdim=True).values
    log_normalizer = torch.log(torch.exp(shifted_logits).sum(dim=-1))
    target_logits = shifted_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


class MultiHeadSelfAttention(nn.Module):
    """不含位置编码的因果多头自注意力层。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将最后一维拆为 (num_heads, head_dim)，并把 head 变为 batch-like 维。"""
        return x.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """执行因果自注意力，禁止每个位置访问未来 token。"""
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        sequence_length = x.shape[-2]
        causal_mask = torch.ones((sequence_length, sequence_length), device=x.device, dtype=torch.bool).tril()
        attention_output = scaled_dot_product_attention(query, key, value, causal_mask)
        merged_heads = attention_output.transpose(-3, -2).flatten(start_dim=-2)
        return self.output_proj(merged_heads)


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """在查询和键投影上应用 RoPE 的因果多头自注意力层。"""

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
        self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """对 Q、K 应用位置旋转；V 不进行位置编码。"""
        sequence_length = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=x.device)

        query = self.rope(self._split_heads(self.q_proj(x)), token_positions)
        key = self.rope(self._split_heads(self.k_proj(x)), token_positions)
        value = self._split_heads(self.v_proj(x))
        causal_mask = torch.ones((sequence_length, sequence_length), device=x.device, dtype=torch.bool).tril()
        attention_output = scaled_dot_product_attention(query, key, value, causal_mask)
        merged_heads = attention_output.transpose(-3, -2).flatten(start_dim=-2)
        return self.output_proj(merged_heads)


class TransformerBlock(nn.Module):
    """由 RoPE 因果注意力和 SwiGLU 组成的 pre-norm Transformer block。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        norm_position: str = "pre",
    ) -> None:
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError("norm_position 必须是 'pre' 或 'post'")
        self.norm_position = norm_position
        # 消融实验中以恒等映射替换全部 RMSNorm；默认结构保持 pre-norm。
        self.ln1: nn.Module = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        if use_rope:
            self.attn: nn.Module = MultiHeadSelfAttentionWithRoPE(
                d_model,
                num_heads,
                max_seq_len,
                theta,
                device=device,
                dtype=dtype,
            )
        else:
            # NoPE 消融：因果 mask 保留，但 Q/K 不注入任何位置编码。
            self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2: nn.Module = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.ffn: nn.Module
        if use_swiglu:
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """按照配置执行 pre-norm 或 post-norm 残差更新。"""
        if self.norm_position == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))

        x = self.ln1(x + self.attn(x, token_positions))
        return self.ln2(x + self.ffn(x))


class TransformerLM(nn.Module):
    """由 token embedding、pre-norm blocks 和词表输出头组成的因果语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        use_swiglu: bool = True,
        norm_position: str = "pre",
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    theta,
                    device=device,
                    dtype=dtype,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    use_swiglu=use_swiglu,
                    norm_position=norm_position,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final: nn.Module = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回每个输入位置预测下一个 token 的未归一化 logits。"""
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError("输入序列长度不能超过 context_length")
        token_positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        return self.lm_head(self.ln_final(x))
