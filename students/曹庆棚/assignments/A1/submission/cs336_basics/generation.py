from __future__ import annotations

import torch
from torch import Tensor

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def sample_token(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled_sorted).squeeze(-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[str, list[int], str]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    model_device = next(model.parameters()).device
    device = torch.device(device) if device is not None else model_device
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must encode to at least one token")
    generated = list(token_ids)
    stop_reason = "max_new_tokens"

    model.eval()
    for _ in range(max_new_tokens):
        context = generated[-model.context_length :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1]
        next_token = int(sample_token(logits, temperature=temperature, top_p=top_p).item())
        generated.append(next_token)
        if end_token_id is not None and next_token == end_token_id:
            stop_reason = "end_token"
            break

    return tokenizer.decode(generated), generated, stop_reason
