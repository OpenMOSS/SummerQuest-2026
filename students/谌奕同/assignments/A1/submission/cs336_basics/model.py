"""Basic model components for assignment 1."""

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class Identity(nn.Module):
    """Identity transformation (no-op), used when an optional norm is disabled."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(nn.Module):
    """A linear layer without bias: y = x @ W^T."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        std = math.sqrt(2.0 / (d_in + d_out))
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_in) -> (..., d_out)
        return x @ self.weight.T

    def load_weights(self, weights: Tensor) -> None:
        """Load externally provided weights."""
        with torch.no_grad():
            self.weight.copy_(weights)


class Embedding(nn.Module):
    """An embedding layer: lookup token IDs in a weight matrix."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (...) -> (... , d_model)
        return self.weight[token_ids]

    def load_weights(self, weights: Tensor) -> None:
        """Load externally provided weights."""
        with torch.no_grad():
            self.weight.copy_(weights)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        output = x / rms * self.weight.to(torch.float32)
        return output.to(in_dtype)

    def load_weights(self, weights: Tensor) -> None:
        """Load externally provided weights."""
        with torch.no_grad():
            self.weight.copy_(weights)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, zero_init_output: bool = False):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        if zero_init_output:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU: silu(x @ W1^T) * (x @ W3^T) @ W2^T
        a = (self.w1(x) * torch.sigmoid(self.w1(x))) * self.w3(x)
        return self.w2(a)

    def load_weights(self, w1: Tensor, w2: Tensor, w3: Tensor) -> None:
        """Load externally provided weights."""
        self.w1.load_weights(w1)
        self.w2.load_weights(w2)
        self.w3.load_weights(w3)


class SiLUFeedForward(nn.Module):
    """SiLU feed-forward network without gating, matching SwiGLU parameter count."""

    def __init__(self, d_model: int, d_ff: int, zero_init_output: bool = False):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        if zero_init_output:
            nn.init.zeros_(self.w2.weight)

    def forward(self, x: Tensor) -> Tensor:
        # SiLU(x @ W1^T) @ W2^T
        a = self.w1(x) * torch.sigmoid(self.w1(x))
        return self.w2(a)

    def load_weights(self, w1: Tensor, w2: Tensor) -> None:
        """Load externally provided weights."""
        self.w1.load_weights(w1)
        self.w2.load_weights(w2)


class RoPE(nn.Module):
    """Rotary Position Embedding using complex-number multiplication."""

    def __init__(self, d_k: int, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires d_k to be even.")
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self._precompute_freqs()

    def _precompute_freqs(self) -> None:
        """Precompute complex rotation frequencies for each dimension pair."""
        # For each pair of dimensions (2i, 2i+1), compute theta^(-2i/d_k)
        dim_indices = torch.arange(0, self.d_k, 2, dtype=torch.float32)
        freqs = 1.0 / (self.theta ** (dim_indices / self.d_k))  # (d_k // 2,)
        # Positions: (max_seq_len,)
        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        # Outer product: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)
        # Complex rotation factors: cos(angle) + i*sin(angle)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        x: Tensor,
        token_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Optional tensor of shape (..., seq_len) with token positions.
                If None, assumes positions [0, 1, ..., seq_len-1].

        Returns:
            Tensor of shape (..., seq_len, d_k) with RoPE applied.
        """
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Expected d_k={self.d_k}, got {d_k}"

        # Select rotation factors for the given positions.
        if token_positions is None:
            freqs_cis = self.freqs_cis[:seq_len]  # (seq_len, d_k//2)
        else:
            freqs_cis = self.freqs_cis[token_positions]  # (..., seq_len, d_k//2)

        # Reshape x to complex view: (..., seq_len, d_k//2)
        x_reshaped = x.reshape(*batch_dims, seq_len, d_k // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)

        # Apply rotation via complex multiplication.
        x_rotated = x_complex * freqs_cis

        # Convert back to real and reshape.
        x_out = torch.view_as_real(x_rotated)  # (..., seq_len, d_k//2, 2)
        return x_out.reshape(*batch_dims, seq_len, d_k)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along dimension `dim`."""
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: (..., queries, d_k)
        K: (..., keys, d_k)
        V: (..., keys, d_v)
        mask: (..., queries, keys) bool tensor. True values are kept.

    Returns:
        (..., queries, d_v)
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        # Match PyTorch convention: True means "attend", False means "mask out".
        scores = scores.masked_fill(~mask, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE and optional QK-Norm."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: Optional[float] = None,
        qk_norm: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qk_norm = qk_norm

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if zero_init_output:
            nn.init.zeros_(self.output_proj.weight)

        self.q_norm = RMSNorm(self.d_head) if qk_norm else None
        self.k_norm = RMSNorm(self.d_head) if qk_norm else None

        self.rope = None
        if theta is not None:
            self.rope = RoPE(self.d_head, theta, max_seq_len)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        token_positions: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project and reshape to (batch, num_heads, seq, d_head).
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        if self.qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # Apply RoPE if enabled.
        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causal mask by default for autoregressive self-attention.
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        # Scaled dot-product attention.
        attn_out = scaled_dot_product_attention(Q, K, V, mask)

        # Reshape back and project.
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.output_proj(attn_out)

    def load_weights(
        self,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        output_proj: Tensor,
    ) -> None:
        self.q_proj.load_weights(q_proj)
        self.k_proj.load_weights(k_proj)
        self.v_proj.load_weights(v_proj)
        self.output_proj.load_weights(output_proj)


class TransformerBlock(nn.Module):
    """Transformer block with configurable norm placement, RoPE, and FFN type."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
        use_rmsnorm: bool = True,
        use_post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        qk_norm: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rmsnorm = use_rmsnorm
        self.use_post_norm = use_post_norm

        norm_cls = RMSNorm if use_rmsnorm else Identity
        self.ln1 = norm_cls(d_model, eps) if use_rmsnorm else Identity()
        self.ln2 = norm_cls(d_model, eps) if use_rmsnorm else Identity()

        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta if use_rope else None,
            qk_norm=qk_norm, zero_init_output=zero_init_output,
        )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, zero_init_output=zero_init_output)
        elif ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model, d_ff, zero_init_output=zero_init_output)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        token_positions: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_post_norm:
            # Post-norm: norm is applied after the residual addition.
            x = self.ln1(x + self.attn(x, mask, token_positions))
            x = self.ln2(x + self.ffn(x))
        else:
            # Pre-norm residual.
            x = x + self.attn(self.ln1(x), mask, token_positions)
            x = x + self.ffn(self.ln2(x))
        return x

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        self.attn.load_weights(
            weights["attn.q_proj.weight"],
            weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],
            weights["attn.output_proj.weight"],
        )
        if self.use_rmsnorm:
            self.ln1.load_weights(weights["ln1.weight"])
            self.ln2.load_weights(weights["ln2.weight"])
        if isinstance(self.ffn, SwiGLU):
            self.ffn.load_weights(
                weights["ffn.w1.weight"],
                weights["ffn.w2.weight"],
                weights["ffn.w3.weight"],
            )
        else:
            self.ffn.load_weights(
                weights["ffn.w1.weight"],
                weights["ffn.w2.weight"],
            )


class TransformerLM(nn.Module):
    """Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        use_rmsnorm: bool = True,
        use_post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        qk_norm: bool = False,
        zero_init_output: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    eps,
                    use_rmsnorm=use_rmsnorm,
                    use_post_norm=use_post_norm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    qk_norm=qk_norm,
                    zero_init_output=zero_init_output,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps) if use_rmsnorm else Identity()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, in_indices: Tensor) -> Tensor:
        batch_size, seq_len = in_indices.shape

        x = self.token_embeddings(in_indices)

        # Causal mask: True means "attend" (lower triangle including diagonal).
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        # Token positions for RoPE.
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq)

        for layer in self.layers:
            x = layer(x, mask, token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def load_weights(self, weights: dict[str, Tensor]) -> None:
        self.token_embeddings.load_weights(weights["token_embeddings.weight"])
        if isinstance(self.ln_final, RMSNorm):
            self.ln_final.load_weights(weights["ln_final.weight"])
        self.lm_head.load_weights(weights["lm_head.weight"])
        for i, layer in enumerate(self.layers):
            layer.load_weights(
                {k.replace(f"layers.{i}.", ""): v for k, v in weights.items() if k.startswith(f"layers.{i}.")}
            )
