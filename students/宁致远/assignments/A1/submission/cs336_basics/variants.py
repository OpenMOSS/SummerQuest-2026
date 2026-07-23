"""Ablation variants of TransformerLM (no RMSNorm, post-norm, NoPE, SiLU FFN)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .model import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RoPE,
    SwiGLU,
    TransformerLM,
    silu,
)


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class SiLUFFN(nn.Module):
    """W2 · silu(W1 x); d_ff·1.5 keeps param count close to SwiGLU's 3·d·d_ff."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        d_ff_eff = int(round(1.5 * d_ff))
        self.w1 = Linear(d_model, d_ff_eff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff_eff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class AblationBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        rope: RoPE | None,
        norm: str,        # rmsnorm | none
        norm_pos: str,    # pre | post
        ffn: str,         # swiglu | silu
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_pos = norm_pos
        mk = (lambda: RMSNorm(d_model, device=device, dtype=dtype)) if norm == "rmsnorm" else Identity
        self.ln1 = mk()
        self.ln2 = mk()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype) if ffn == "swiglu" else SiLUFFN(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        if self.norm_pos == "pre":
            x = x + self.attn(self.ln1(x), positions)
            x = x + self.ffn(self.ln2(x))
        else:  # post-norm
            x = self.ln1(x + self.attn(x, positions))
            x = self.ln2(x + self.ffn(x))
        return x


class AblationLM(nn.Module):
    def __init__(self, cfg: dict, device=None, dtype=None):
        super().__init__()
        self.context_length = cfg["context_length"]
        d_model = cfg["d_model"]
        self.token_embeddings = Embedding(cfg["vocab_size"], d_model, device=device, dtype=dtype)
        use_rope = cfg["use_rope"]
        rope = RoPE(cfg["rope_theta"], d_model // cfg["num_heads"], cfg["context_length"], device=device) if use_rope else None
        self.layers = nn.ModuleList([
            AblationBlock(
                d_model, cfg["num_heads"], cfg["d_ff"], cfg["context_length"], cfg["rope_theta"],
                rope=rope, norm=cfg["norm"], norm_pos=cfg["norm_pos"], ffn=cfg["ffn"],
                device=device, dtype=dtype,
            )
            for _ in range(cfg["num_layers"])
        ])
        if cfg["norm"] == "rmsnorm":
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = Identity()
        self.lm_head = Linear(d_model, cfg["vocab_size"], device=device, dtype=dtype)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.token_embeddings(ids)
        positions = torch.arange(ids.size(-1), device=ids.device).expand_as(ids)
        for blk in self.layers:
            x = blk(x, positions)
        x = self.ln_final(x)
        return self.lm_head(x)


def build_variant(name: str, args, dtype) -> nn.Module:
    base = dict(
        vocab_size=args.vocab_size, context_length=args.context_length,
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=args.rope_theta,
        norm="rmsnorm", norm_pos="pre", ffn="swiglu", use_rope=True,
    )
    if name == "no_rmsnorm":
        base["norm"] = "none"
    elif name == "post_norm":
        base["norm_pos"] = "post"
    elif name == "nope":
        base["use_rope"] = False
    elif name == "silu_ffn":
        base["ffn"] = "silu"
    else:
        raise ValueError(f"unknown variant: {name}")
    return AblationLM(base, device=args.device, dtype=dtype)
