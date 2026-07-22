"""Autoregressive decoding with temperature and nucleus (top-p) sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch import Tensor, nn


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token IDs from a final vocabulary dimension."""

    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if temperature == 0:
        return logits.argmax(dim=-1)

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # Retain the smallest prefix whose cumulative probability reaches top_p.
        keep = cumulative - sorted_probabilities < top_p
        sorted_probabilities = sorted_probabilities * keep
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_sorted_index = torch.multinomial(
            sorted_probabilities.reshape(-1, sorted_probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(*logits.shape[:-1], 1)
        return sorted_indices.gather(dim=-1, index=sampled_sorted_index).squeeze(-1)

    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1, generator=generator
    ).reshape(logits.shape[:-1])


@torch.inference_mode()
def generate_ids(
    model: nn.Module,
    prompt_ids: Sequence[int] | Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_id: int | None = None,
    context_length: int | None = None,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Generate token IDs autoregressively from ``prompt_ids``."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if isinstance(prompt_ids, Tensor):
        tokens = prompt_ids.detach().long().flatten().tolist()
    else:
        tokens = [int(token_id) for token_id in prompt_ids]
    if not tokens:
        raise ValueError("prompt_ids must not be empty")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    if context_length is None:
        context_length = getattr(model, "context_length", None)

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            model_input = tokens[-context_length:] if context_length is not None else tokens
            input_tensor = torch.tensor(model_input, dtype=torch.long, device=device).unsqueeze(0)
            next_logits = model(input_tensor)[0, -1]
            next_token = int(
                sample_next_token(next_logits, temperature=temperature, top_p=top_p, generator=generator).item()
            )
            tokens.append(next_token)
            if end_of_text_id is not None and next_token == end_of_text_id:
                break
    finally:
        model.train(was_training)
    return tokens


def generate(
    model: nn.Module,
    tokenizer: TokenizerLike,
    prompt: str,
    max_new_tokens: int,
    **kwargs,
) -> str:
    """Encode a text prompt, generate tokens, and decode the full sequence."""

    token_ids = generate_ids(model, tokenizer.encode(prompt), max_new_tokens, **kwargs)
    return tokenizer.decode(token_ids)
