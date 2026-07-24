from __future__ import annotations

import torch

from .nn_utils import softmax


def _pick_one(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    # 先把大概率的排前面，后面那截不够 top_p 的直接扔掉。
    sorted_p, sorted_id = probabilities.sort(dim=-1, descending=True)
    keep_it = sorted_p.cumsum(dim=-1) - sorted_p < top_p
    sorted_p = torch.where(keep_it, sorted_p, torch.zeros_like(sorted_p))
    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
    picked_rank = torch.multinomial(sorted_p, num_samples=1)
    return sorted_id.gather(-1, picked_rank)


@torch.no_grad()
def generate(
    model,
    prompt_ids: list[int],
    eos_id: int | None,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    context_length: int | None = None,
) -> list[int]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty")

    device = next(model.parameters()).device
    all_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        model_in = all_ids if context_length is None else all_ids[:, -context_length:]
        last_logits = model(model_in)[:, -1] / temperature
        next_one = _pick_one(softmax(last_logits, dim=-1), top_p)
        all_ids = torch.cat((all_ids, next_one), dim=-1)
        if eos_id is not None and next_one.item() == eos_id:
            break
    return all_ids.squeeze(0).tolist()
