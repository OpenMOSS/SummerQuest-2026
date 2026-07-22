"""Decoder-only Transformer language model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention, scaled_dot_product_attention
from .nn import Embedding, Identity, Linear, RMSNorm, SiLUFeedForward, SwiGLU, silu
from .rope import RoPE, RotaryPositionalEmbedding
from .transformer import TransformerBlock


@dataclass(slots=True)
class TransformerLMConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10_000.0
    remove_rmsnorm: bool = False
    use_post_norm: bool = False
    remove_rope: bool = False
    ffn_type: str = "swiglu"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TransformerLM(nn.Module):
    """A pre-norm causal Transformer LM, with configurable ablations."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        norm_mode: str | None = None,
        use_rope: bool | None = None,
        ffn_type: str | None = "swiglu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or d_model <= 0 or num_layers <= 0:
            raise ValueError("vocab_size, context_length, d_model, and num_layers must be positive")
        if norm_mode is not None:
            norm_mode = norm_mode.lower()
            if norm_mode not in {"pre", "post", "none"}:
                raise ValueError("norm_mode must be 'pre', 'post', or 'none'")
            remove_rmsnorm = norm_mode == "none"
            use_post_norm = norm_mode == "post"
        if use_rope is not None:
            remove_rope = not use_rope
        normalized_ffn_type = "swiglu" if ffn_type is None else ffn_type

        self.config = TransformerLMConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            remove_rmsnorm=remove_rmsnorm,
            use_post_norm=use_post_norm,
            remove_rope=remove_rope,
            ffn_type=normalized_ffn_type,
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    remove_rmsnorm=remove_rmsnorm,
                    use_post_norm=use_post_norm,
                    remove_rope=remove_rope,
                    ffn_type=normalized_ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = Identity() if remove_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    @property
    def context_length(self) -> int:
        return self.config.context_length

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if token_ids.ndim < 1:
            raise ValueError("token_ids must have at least one dimension")
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"sequence length {sequence_length} exceeds context length {self.context_length}")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=token_ids.device)

        hidden = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden = layer(hidden, token_positions)
        return self.lm_head(self.ln_final(hidden))

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressively sample tokens, cropping inputs to the context window."""

        if token_ids.ndim != 2:
            raise ValueError("generation input must have shape (batch, sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        generated = token_ids
        finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
        for _ in range(max_new_tokens):
            window = generated[:, -self.context_length :]
            logits = self(window)[:, -1, :] / temperature
            probabilities = torch.softmax(logits.float(), dim=-1)
            if top_p is not None and top_p < 1:
                sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
                cumulative = sorted_probabilities.cumsum(dim=-1)
                remove = cumulative - sorted_probabilities >= top_p
                sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
                sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
                sampled_sorted = torch.multinomial(sorted_probabilities, num_samples=1)
                next_token = sorted_indices.gather(-1, sampled_sorted)
            else:
                next_token = torch.multinomial(probabilities, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(finished[:, None], eos_token_id, next_token)
                finished |= next_token.squeeze(-1).eq(eos_token_id)
            generated = torch.cat((generated, next_token), dim=-1)
            if bool(finished.all()):
                break
        return generated


__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RoPE",
    "RotaryPositionalEmbedding",
    "SiLUFeedForward",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "TransformerLMConfig",
    "scaled_dot_product_attention",
    "silu",
]
