from __future__ import annotations

import torch
from torch import Tensor

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import softmax


def sample_top_p(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    probabilities = softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
    sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
    sampled_sorted_index = torch.multinomial(sorted_probabilities, num_samples=1)
    return sorted_indices.gather(dim=-1, index=sampled_sorted_index).squeeze(-1)


@torch.inference_mode()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    device: str | torch.device = "cpu",
) -> list[int]:
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    generated = list(prompt_ids)
    model.eval()
    for _ in range(max_new_tokens):
        context = generated[-model.context_length :]
        inputs = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        next_token = int(sample_top_p(model(inputs)[:, -1], temperature, top_p).item())
        generated.append(next_token)
        if end_token_id is not None and next_token == end_token_id:
            break
    return generated
