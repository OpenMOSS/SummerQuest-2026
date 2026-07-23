"""Autoregressive text generation with temperature and top-p (nucleus) sampling."""

from __future__ import annotations

import torch

from .model import TransformerLM, softmax
from .tokenizer import Tokenizer


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token: str = "<|endoftext|>",
    device: str = "cpu",
) -> str:
    """Generate a continuation for ``prompt``.

    Stops at ``max_new_tokens`` or when the end-of-text token is sampled.
    """
    model.eval()
    context_length = model.context_length
    ids = tokenizer.encode(prompt) if prompt else []
    eot_id = tokenizer.special_ids.get(eot_token)

    generated: list[int] = []
    for _ in range(max_new_tokens):
        window = ids[-context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1]  # (vocab_size,)

        if temperature == 0.0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / temperature
            probs = softmax(logits, dim=-1)
            if top_p < 1.0:
                probs = _top_p_filter(probs, top_p)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if eot_id is not None and next_id == eot_id:
            break
        ids.append(next_id)
        generated.append(next_id)

    return tokenizer.decode(generated)


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative prob >= top_p, renormalize."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep tokens up to and including the one that crosses the top_p threshold.
    keep = cumulative - sorted_probs < top_p
    keep[0] = True
    filtered = torch.zeros_like(probs)
    filtered[sorted_idx[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()
