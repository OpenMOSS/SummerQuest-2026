"""Autoregressive decoding with temperature + top-p."""

from __future__ import annotations

import torch
from torch import Tensor

from .model import TransformerLM, softmax


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_id: int | None = None,
    device: str = "cpu",
) -> list[int]:
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    ctx = model.context_length
    out: list[int] = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = ids[:, -ctx:]
        logits = model(x)[:, -1, :]
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        probs = softmax(logits, dim=-1)
        if 0 < top_p < 1.0:
            sp, si = probs.sort(dim=-1, descending=True)
            cum = sp.cumsum(dim=-1)
            keep = cum - sp <= top_p  # keep smallest set whose cumulative prob >= top_p
            sp = sp * keep
            sp = sp / sp.sum(-1, keepdim=True)
            nxt_sorted = torch.multinomial(sp, num_samples=1)
            nxt = si.gather(-1, nxt_sorted)
        else:
            nxt = torch.multinomial(probs, num_samples=1)
        tok = int(nxt.item())
        out.append(tok)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and tok == eos_id:
            break
    return out
