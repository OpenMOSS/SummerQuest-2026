from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Module

from cs336_basics.model import softmax
from cs336_basics.tokenizer import BPETokenizer


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    generated_token_ids: list[int]
    text: str


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token using temperature scaling and nucleus sampling."""
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    probabilities = softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        keep_sorted = cumulative_probabilities - sorted_probabilities < top_p
        filtered_sorted = torch.where(keep_sorted, sorted_probabilities, torch.zeros_like(sorted_probabilities))
        filtered_sorted = filtered_sorted / torch.sum(filtered_sorted)
        sampled_sorted_index = torch.multinomial(filtered_sorted, num_samples=1, generator=generator)
        return int(sorted_indices[sampled_sorted_index].item())

    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


@torch.no_grad()
def generate_token_ids(
    model: Module,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively extend a non-empty prompt and return only newly generated IDs."""
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must contain at least one token.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")

    resolved_device = _model_device(model) if device is None else torch.device(device)
    context_length = getattr(model, "context_length", None)
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("model must expose a positive integer context_length.")

    was_training = model.training
    model.eval()
    all_token_ids = list(prompt_token_ids)
    generated_token_ids: list[int] = []
    try:
        for _ in range(max_new_tokens):
            model_input = torch.tensor(
                [all_token_ids[-context_length:]],
                dtype=torch.long,
                device=resolved_device,
            )
            next_token_logits = model(model_input)[0, -1]
            next_token_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            all_token_ids.append(next_token_id)
            generated_token_ids.append(next_token_id)
            if eos_token_id is not None and next_token_id == eos_token_id:
                break
    finally:
        model.train(was_training)
    return generated_token_ids


def generate_text(
    model: Module,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    prompt_token_ids = tokenizer.encode(prompt)
    generated_token_ids = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        generator=generator,
    )
    token_ids = prompt_token_ids + generated_token_ids
    return GenerationResult(
        token_ids=token_ids,
        generated_token_ids=generated_token_ids,
        text=tokenizer.decode(token_ids),
    )


def _model_device(model: Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
