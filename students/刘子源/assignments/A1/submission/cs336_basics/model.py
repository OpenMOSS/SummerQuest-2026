# 这里故意没用现成 Linear/Embedding，作业就要看手搓版，别给它换回去。

from __future__ import annotations

import math

import torch
from torch import nn

from .nn_utils import softmax


@torch.no_grad()
def _truncated_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 0.02,
    lower: float = -0.06,
    upper: float = 0.06,
) -> torch.Tensor:
    # 这坨是 inverse-CDF 初始化，看着绕，但数值快照就认这个算法。
    def get_normal_cdf(value: float) -> float:
        return (1.0 + math.erf(value / math.sqrt(2.0))) / 2.0

    cdf_left = get_normal_cdf((lower - mean) / std)
    cdf_right = get_normal_cdf((upper - mean) / std)
    tensor.uniform_(2 * cdf_left - 1, 2 * cdf_right - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0)).add_(mean).clamp_(min=lower, max=upper)
    return tensor


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # no-rmsnorm 实验用的空壳，真就啥也不干。
        return x


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        _truncated_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # weight 存的是 out × in，所以这里必须转一下，别顺手删。
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _truncated_normal_(self.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_dtype = x.dtype
        x_32 = x.float()  # 半精度直接算平方容易飘，先升一下再说。
        rms_num = torch.sqrt(x_32.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_32 / rms_num * self.weight.float()).to(old_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        freq_back = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        pos_list = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angle_table = pos_list[:, None] * freq_back[None, :]
        self.register_buffer("cos", angle_table.cos(), persistent=False)
        self.register_buffer("sin", angle_table.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if token_positions.max().item() >= self.cos.shape[0]:
            raise ValueError("token position exceeds the configured maximum sequence length")
        cos = self.cos[token_positions].to(device=x.device, dtype=x.dtype)
        sin = self.sin[token_positions].to(device=x.device, dtype=x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        turned = torch.empty_like(x)
        turned[..., 0::2] = x_even * cos - x_odd * sin
        turned[..., 1::2] = x_even * sin + x_odd * cos
        return turned


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    score = queries @ keys.transpose(-1, -2) / math.sqrt(queries.shape[-1])
    if mask is not None:
        # mask 里 False 的位置就是“不准偷看”，直接塞成负无穷。
        score = score.masked_fill(~mask, float("-inf"))
    return softmax(score, dim=-1) @ values


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
        else:
            self.rope = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        split_shape = (*x.shape[:-2], seq_len, self.num_heads, self.head_dim)
        return x.reshape(split_shape).transpose(-3, -2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        x = x.transpose(-3, -2).contiguous()
        # transpose 后不是连续内存，contiguous 少了 reshape 会闹脾气。
        return x.reshape(*x.shape[:-3], seq_len, self.d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        can_see = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        mixed = scaled_dot_product_attention(q, k, v, can_see)
        return self.output_proj(self._combine_heads(mixed))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_kind: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.post_norm = post_norm
        self.attn = MultiheadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope=use_rope, device=device, dtype=dtype
        )
        # 就两种情况，写 lambda 工厂纯属把简单事整复杂了。
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln1 = Identity()
            self.ln2 = Identity()
        if ffn_kind == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_kind == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown ffn_kind: {ffn_kind}")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.post_norm:
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_kind: str = "swiglu",
        device=None,
        dtype=None,
    ):
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
                    rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    post_norm=post_norm,
                    use_rope=use_rope,
                    ffn_kind=ffn_kind,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        if use_rmsnorm:
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("input sequence exceeds context_length")
        pos = torch.arange(seq_len, device=token_ids.device)
        hidden = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden = layer(hidden, pos)
        return self.lm_head(self.ln_final(hidden))
