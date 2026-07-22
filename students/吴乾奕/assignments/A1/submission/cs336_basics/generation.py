"""Autoregressive decoding with temperature and nucleus sampling."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from .losses import softmax


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token IDs from the final dimension of a logits tensor.

    ``temperature=0`` is treated as deterministic greedy decoding. For
    positive temperatures, top-p keeps the smallest descending-probability set
    whose mass reaches ``top_p`` and renormalizes before sampling.
    """

    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")

    if temperature == 0:
        return logits.argmax(dim=-1)

    working_logits = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    probabilities = softmax(working_logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        # Retain the first item crossing the threshold as well as everything
        # before it. This also guarantees that at least one item is retained.
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(
            sorted_probabilities.reshape(-1, sorted_probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(sorted_probabilities.shape[:-1])
        return sorted_indices.gather(-1, sampled_sorted.unsqueeze(-1)).squeeze(-1)

    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        num_samples=1,
        generator=generator,
    ).reshape(probabilities.shape[:-1])


@torch.inference_mode()
def generate_token_ids(
    model: torch.nn.Module,
    prompt_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
    device: str | torch.device,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    seed: int | None = None,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> list[int]:
    """Generate and return prompt plus newly sampled token IDs."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    target_device = torch.device(device)
    if isinstance(prompt_ids, Tensor):
        ids = prompt_ids.detach().to(device=target_device, dtype=torch.long).flatten().tolist()
    else:
        ids = [int(token_id) for token_id in prompt_ids]
    if not ids:
        if eos_token_id is None:
            raise ValueError("an empty prompt requires eos_token_id as a beginning token")
        ids = [int(eos_token_id)]

    generator = None
    if seed is not None and target_device.type in {"cpu", "cuda"}:
        generator = torch.Generator(device=target_device)
    if generator is not None:
        generator.manual_seed(seed)
    elif seed is not None:
        torch.manual_seed(seed)

    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            model_input = torch.tensor(ids[-context_length:], dtype=torch.long, device=target_device).unsqueeze(0)
            amp_context = (
                torch.autocast(device_type=target_device.type, dtype=amp_dtype)
                if amp_enabled and amp_dtype != torch.float32
                else nullcontext()
            )
            with amp_context:
                logits = model(model_input)
            next_id = int(
                sample_next_token(
                    logits[0, -1],
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                ).item()
            )
            ids.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
    finally:
        model.train(was_training)
    return ids


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    **generation_kwargs: Any,
) -> tuple[str, list[int]]:
    """Encode a prompt, generate token IDs, and decode the complete sequence."""

    prompt_ids = tokenizer.encode(prompt)
    token_ids = generate_token_ids(model, prompt_ids, **generation_kwargs)
    return tokenizer.decode(token_ids), token_ids
