import torch
import torch.nn as nn
import math
from typing import Literal

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        parameter = torch.empty((out_features, in_features), dtype=dtype, device=device)
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(parameter, std=sigma, a=-3*sigma, b=3*sigma)
        self.weight = nn.Parameter(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        parameter = torch.empty((num_embeddings, embedding_dim), dtype=dtype, device=device)
        nn.init.trunc_normal_(parameter, a=-3, b=3)
        self.weight = nn.Parameter(parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        parameter = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(parameter) # (d_model, )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32) # (..., d_model)
        inverse_rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) # (..., 1)
        normalized = x * inverse_rms
        y = normalized * self.weight
        return y.to(input_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    gate = torch.sigmoid(x)
    return x * gate

class SwiGLU(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x               (..., d_model)
        w1(x)           (..., d_ff)
        silu(w1(x))     (..., d_ff) <- gate
        w3(x)           (..., d_ff) <- value
        gate * value    (..., d_ff)
        w2(..)          (..., d_model)
        """
        gate = silu(self.w1(x))
        value = self.w3(x)
        hidden = gate * value
        return self.w2(hidden)

class SiLUFFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = silu(self.w1(x))
        return self.w2(hidden)

class RoPE(nn.Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(
            self,
            d_k: int,
            theta: float,
            max_seq_len: int,
            device: torch.device | None = None,
    ) -> None:
        r"""Initialize RoPE frequency caches.

        For each channel pair ``(2k, 2k+1)`` we compute an inverse frequency
        ``\omega_k = 1 / theta^{2k/d_k}`` (in radians per token).  The cosine and
        sine caches are precomputed for all positions up to ``max_seq_len``.
        """
        super().__init__()
        dimension_indices = torch.arange(0, d_k, 2, device=device)      # (d_k / 2, )
        inverse_frequencies = 1.0 / (
            theta ** (dimension_indices / d_k)
        )
        positions = torch.arange(max_seq_len, device=device)            # (max_seq_len, )
        angles = positions[:, None] * inverse_frequencies[None, :]      # (max_seq_len, d_k / 2)

        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions  (..., seq_len)
        cos = self.cos_cache[token_positions] # (..., seq_len, d_k / 2)
        sin = self.sin_cache[token_positions] # (..., seq_len, d_k / 2)

        # x                (..., seq_len, d_k)
        x_even = x[..., 0::2] # (..., seq_len, d_k / 2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k / 2)

        rotated_even = x_even * cos - x_odd * sin   # (..., seq_len, d_k / 2)
        rotated_odd = x_even * sin + x_odd * cos    # (..., seq_len, d_k / 2)

        paired = torch.stack((rotated_even, rotated_odd), dim=-1) # (..., seq_len, d_k / 2, 2)
        return paired.flatten(start_dim = -2) # (..., seq_len, d_k)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    maximum = torch.amax(x, dim=dim, keepdim=True)              # (..., 1, ...)
    shifted = x - maximum
    numerator = torch.exp(shifted)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Q:      (..., queries, d_k)
    K:      (..., keys,    d_k)
    V:      (..., keys,    d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2,-1)
    scores = scores / math.sqrt(d_k) # scaled。除以 \sqrt{d_k} 后，score 的标准差重新回到 ~1

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = softmax(scores, dim=-1)
    return weights @ V

class MHA(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            rope: RoPE | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q = self.q_proj(x) # (..., seq, d_model)
        q = q.unflatten(-1, (self.n_heads, self.d_head)) # (..., seq, n_heads, d_head)
        q = q.transpose(-3, -2) # (..., n_heads, seq, d_head), n_heads 变成一个 batch-like 的维度

        k = self.k_proj(x)
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        k = k.transpose(-3, -2)

        v = self.v_proj(x)
        v = v.unflatten(-1, (self.n_heads, self.d_head))
        v = v.transpose(-3, -2)

        """
        构造 causal mask. Causal mask 的行是 query, 列是 key:
        T F F F
        T T F F
        T T T F
        T T T T
        """
        seq_len = x.shape[-2]
        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        # Apply RoPE
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    x.shape[-2],
                    device=x.device,
                )

            positions_for_heads = token_positions.unsqueeze(-2) # (..., 1, seq)
            q = self.rope(q, positions_for_heads)
            k = self.rope(k, positions_for_heads)

        context = scaled_dot_product_attention(q, k, v, causal_mask) # (..., n_heads, seq, d_head)

        context = context.transpose(-3, -2) # (..., seq, n_heads, d_head)
        context = context.flatten(start_dim=-2) # (..., seq, d_model)
        return self.output_proj(context)

class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            norm_mode: Literal["pre", "post", "none"] = "pre",
            position_mode: Literal["rope", "none"] = "rope",
            ffn_mode: Literal["swiglu", "silu"] = "swiglu",
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if norm_mode not in ("pre", "post", "none"):
            raise ValueError(f"unknown norm mode: {norm_mode}")
        self.norm_mode = norm_mode
        if self.norm_mode != "none":
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        if position_mode not in ("rope", "none"):
            raise ValueError(f"unknown position mode: {position_mode}")
        self.position_mode = position_mode
        if self.position_mode == "rope":
            rope = RoPE(
                d_model // num_heads,
                theta,
                max_seq_len,
                device=device
            )
        else:
            rope = None

        if ffn_mode not in ("swiglu", "silu"):
            raise ValueError(f"unknown ffn mode: {ffn_mode}")
        self.ffn_mode = ffn_mode
        if self.ffn_mode == "swiglu":
            self.ffn = SwiGLU(
                d_model,
                d_ff,
                device=device,
                dtype=dtype
            )
        else:
            self.ffn = SiLUFFN(
                d_model,
                d_ff,
                device=device,
                dtype=dtype
            )

        self.attn = MHA(
            d_model,
            num_heads,
            device=device,
            dtype=dtype,
            rope=rope
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_mode == "pre":
        # pre-norm
            attn_output = self.attn(
                self.ln1(x),
                token_positions
            )
            y = x + attn_output
            ffn_output = self.ffn(self.ln2(y))
            z = y + ffn_output
        elif self.norm_mode == "post":
            attn_output = self.attn(
                x, token_positions
            )
            y = self.ln1(x + attn_output)
            ffn_output = self.ffn(y)
            z = self.ln2(y + ffn_output)
        else:
            attn_output = self.attn(
                x, token_positions
            )
            y = x + attn_output
            ffn_output = self.ffn(y)
            z = y + ffn_output
        return z

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
            norm_mode: Literal["pre", "post", "none"] = "pre",
            position_mode: Literal["rope", "none"] = "rope",
            ffn_mode: Literal["swiglu", "silu"] = "swiglu",
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.position_mode = position_mode
        self.ffn_mode = ffn_mode

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                context_length,
                rope_theta,
                norm_mode=norm_mode,
                position_mode=position_mode,
                ffn_mode=ffn_mode,
                device=device,
                dtype=dtype,
            ) for _ in range(num_layers)
        ])
        if norm_mode == "none":
            self.ln_final = None
        else:
            self.ln_final = RMSNorm(
                d_model,
                device=device,
                dtype=dtype
            )

        self.lm_head = Linear(
            d_model,
            vocab_size,
            device=device,
            dtype=dtype
        )

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(in_indices)

        if self.position_mode == "rope":
            token_positions = torch.arange(
                in_indices.shape[-1],
                device=in_indices.device
            )
        else:
            token_positions = None

        for layer in self.layers:
            x = layer(x, token_positions)

        if self.ln_final is not None:
            x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (..., C)
    targets: (...)
    """
    max_logits = logits.amax(dim=-1, keepdim=True)
    shifted = logits - max_logits
    selected_shifted = shifted.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    loss = log_sum_exp - selected_shifted
    return loss.mean()

if __name__ == "__main__":
    layer = Linear(3,2)
    x = torch.randn(4,5,3)

    assert layer.weight.shape == (2,3)
    assert layer(x).shape == (4,5,2)
    assert torch.allclose(layer(x), x@layer.weight.T)

    # 梯度累加小实验
    embedding = Embedding(5,4)
    token_ids = torch.tensor([[2,0],[1,2]]) # 2 出现两次，0,1 出现一次
    output = embedding(token_ids)
    output.sum().backward()
    print(embedding.weight.grad) # 第二行全是 2, 第 0，1 行全 1，第 3，4 行全 0
