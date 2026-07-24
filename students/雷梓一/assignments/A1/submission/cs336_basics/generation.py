from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import softmax


def sample_next_token(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    probabilities = softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        keep = cumulative - sorted_probabilities < top_p
        filtered = torch.where(keep, sorted_probabilities, torch.zeros_like(sorted_probabilities))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        sampled_rank = torch.multinomial(filtered, num_samples=1)
        return sorted_indices.gather(-1, sampled_rank).squeeze(-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    end_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tensor:
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
        squeeze_batch = True
    elif prompt_ids.ndim == 2:
        squeeze_batch = False
    else:
        raise ValueError("prompt_ids must have shape (sequence,) or (batch, sequence)")
    generated = prompt_ids
    context_length = getattr(model, "context_length", generated.shape[-1] + max_new_tokens)
    was_training = model.training
    model.eval()
    finished = torch.zeros(generated.shape[0], device=generated.device, dtype=torch.bool)
    for _ in range(max_new_tokens):
        model_input = generated[:, -context_length:]
        logits = model(model_input)[:, -1, :]
        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        if end_token_id is not None:
            next_token = torch.where(finished, torch.full_like(next_token, end_token_id), next_token)
        generated = torch.cat((generated, next_token.unsqueeze(-1)), dim=-1)
        if end_token_id is not None:
            finished |= next_token == end_token_id
            if bool(torch.all(finished)):
                break
    if was_training:
        model.train()
    return generated.squeeze(0) if squeeze_batch else generated
