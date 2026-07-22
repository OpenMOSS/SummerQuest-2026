import torch
import torch.nn as nn
import math
from einops import rearrange
from typing import Literal

class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.transpose(-1, -2))
    
class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.eps = eps        
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = (x_fp32 / rms) * self.weight.to(torch.float32)     
        return result.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int | None = None,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
                    
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        w1_x = self.w1(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        return self.w2(silu_w1_x * self.w3(x))

class SiLUFFN(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE."
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = -x[..., 1::2]
        x_rotated[..., 1::2] = x[..., 0::2]
        return x_rotated

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.long()
        cos_x = self.cos[token_positions].to(x.dtype)
        sin_x = self.sin[token_positions].to(x.dtype)
        return x * cos_x + self._rotate_half(x) * sin_x
    
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    shifted_x = x - max_val
    exp_x = torch.exp(shifted_x)    
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp

def scaled_dot_product_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = torch.einsum("... i d, ... j d -> ... i j", q, k)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    attn_probs = softmax(scores, dim=-1)
    output = torch.einsum("... i j, ... j d -> ... i d", attn_probs, v)
    return output

class CausalSelfAttention(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self, 
        x: torch.Tensor, 
        rope: nn.Module | None = None,
        token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        
        q = rearrange(self.w_q(x), "... t (h d) -> ... h t d", h=self.num_heads)
        k = rearrange(self.w_k(x), "... t (h d) -> ... h t d", h=self.num_heads)
        v = rearrange(self.w_v(x), "... t (h d) -> ... h t d", h=self.num_heads)
        
        if rope is not None and token_positions is not None:
            rope_positions = token_positions
            while rope_positions.ndim < q.ndim - 1:
                rope_positions = rope_positions.unsqueeze(-2)
            q = rope(q, rope_positions)
            k = rope(k, rope_positions)
            
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool)
        )
        
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        concat_out = rearrange(attn_out, "... h t d -> ... t (h d)")
        
        return self.w_o(concat_out)
    
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_position: Literal["pre", "post", "none"] = "pre",
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
    ):
        super().__init__()

        if norm_position not in {"pre", "post", "none"}:
            raise ValueError("norm_position must be 'pre', 'post', or 'none'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.norm_position = norm_position
        norm_factory = RMSNorm if norm_position != "none" else None
        self.attn_norm = (
            norm_factory(d_model=d_model, device=device, dtype=dtype) if norm_factory is not None else nn.Identity()
        )
        self.attn = CausalSelfAttention(
            d_model=d_model, num_heads=num_heads, device=device, dtype=dtype
        )

        self.ffn_norm = (
            norm_factory(d_model=d_model, device=device, dtype=dtype) if norm_factory is not None else nn.Identity()
        )
        ffn_class = SwiGLU if ffn_type == "swiglu" else SiLUFFN
        self.ffn = ffn_class(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        rope: nn.Module | None,
        token_positions: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.norm_position == "pre":
            y = x + self.attn(self.attn_norm(x), rope=rope, token_positions=token_positions)
            return y + self.ffn(self.ffn_norm(y))
        if self.norm_position == "post":
            y = self.attn_norm(x + self.attn(x, rope=rope, token_positions=token_positions))
            return self.ffn_norm(y + self.ffn(y))
        y = x + self.attn(x, rope=rope, token_positions=token_positions)
        return y + self.ffn(y)
    
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
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_position: Literal["pre", "post", "none"] = "pre",
        position_encoding: Literal["rope", "none"] = "rope",
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
    ):
        super().__init__()

        if position_encoding not in {"rope", "none"}:
            raise ValueError("position_encoding must be 'rope' or 'none'")
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm_position = norm_position
        self.position_encoding = position_encoding
        self.ffn_type = ffn_type
        
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                norm_position=norm_position,
                ffn_type=ffn_type,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        self.ln_final = (
            RMSNorm(d_model=d_model, device=device, dtype=dtype)
            if norm_position != "none"
            else nn.Identity()
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)        
        d_k = d_model // num_heads
        self.rope = (
            RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=d_k,
                max_seq_len=context_length,
                device=device,
            )
            if position_encoding == "rope"
            else None
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        h = self.token_embeddings(token_ids)
        
        token_positions = (
            torch.arange(seq_len, device=token_ids.device).expand(batch_size, seq_len)
            if self.rope is not None
            else None
        )
        
        for layer in self.layers:
            h = layer(h, rope=self.rope, token_positions=token_positions)
            
        h = self.ln_final(h)        
        logits = self.lm_head(h)
        return logits
