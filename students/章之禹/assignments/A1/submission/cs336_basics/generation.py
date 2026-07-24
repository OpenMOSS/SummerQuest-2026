"""Autoregressive decoding with temperature and nucleus (top-p) sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor


def _probabilities(logits: Tensor, temperature: float) -> Tensor:
    # Shift before dividing so a tiny positive temperature cannot overflow the
    # largest finite logit to +inf and subsequently produce ``inf - inf``.
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    scaled = shifted / temperature
    weights = torch.exp(scaled)
    return weights / weights.sum(dim=-1, keepdim=True)


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample token IDs from a batch of final-position logits.

    ``temperature=0`` selects greedily.  For positive temperatures, top-p
    retains the smallest descending-probability prefix whose cumulative mass
    reaches ``top_p``, then renormalizes implicitly through multinomial
    sampling.
    """

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
        squeeze = True
    elif logits.ndim == 2:
        squeeze = False
    else:
        raise ValueError("logits must have shape (vocab_size,) or (batch_size, vocab_size)")
    if logits.shape[-1] == 0:
        raise ValueError("vocabulary dimension must be non-empty")
    if temperature < 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("temperature must be finite and non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must all be finite")

    if temperature == 0:
        sampled = logits.argmax(dim=-1)
    else:
        probabilities = _probabilities(logits, temperature)
        sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        # Subtracting the current probability shifts the cumulative mass right,
        # so the first token that crosses p remains in the candidate set.
        keep = cumulative - sorted_probabilities < top_p
        filtered_probabilities = sorted_probabilities * keep
        sampled_sorted_index = torch.multinomial(filtered_probabilities, num_samples=1, generator=generator)
        sampled = sorted_indices.gather(dim=-1, index=sampled_sorted_index).squeeze(-1)
    return sampled.squeeze(0) if squeeze else sampled


def _model_context_length(model: torch.nn.Module) -> int | None:
    for name in ("context_length", "max_seq_len", "max_sequence_length"):
        value = getattr(model, name, None)
        if value is not None:
            return int(value)
    return None


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    context_length: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively append up to ``max_new_tokens`` to one or more prompts."""

    if input_ids.ndim == 1:
        output = input_ids.unsqueeze(0).to(dtype=torch.long)
        squeeze = True
    elif input_ids.ndim == 2:
        output = input_ids.to(dtype=torch.long)
        squeeze = False
    else:
        raise ValueError("input_ids must have shape (sequence,) or (batch, sequence)")
    if output.shape[-1] == 0:
        raise ValueError("the prompt must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    limit = context_length if context_length is not None else _model_context_length(model)
    if limit is not None and limit <= 0:
        raise ValueError("context_length must be positive")

    was_training = model.training
    model.eval()
    finished = torch.zeros(output.shape[0], dtype=torch.bool, device=output.device)
    try:
        for _ in range(max_new_tokens):
            model_input = output[:, -limit:] if limit is not None else output
            logits = model(model_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            if logits.ndim != 3 or logits.shape[:2] != model_input.shape:
                raise ValueError(
                    "model must return logits shaped (batch, sequence, vocab); "
                    f"got {tuple(logits.shape)} for input {tuple(model_input.shape)}"
                )
            next_token = sample_next_token(
                logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            if eos_token_id is not None:
                eos = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(finished, eos, next_token)
                finished |= next_token == eos_token_id
            output = torch.cat((output, next_token.unsqueeze(-1)), dim=-1)
            if eos_token_id is not None and bool(finished.all()):
                break
    finally:
        model.train(was_training)
    return output.squeeze(0) if squeeze else output


def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    context_length: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[str, list[int]]:
    """Tokenize a prompt, generate, and decode the complete sequence."""

    prompt_ids: Sequence[int] = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt encoded to an empty token sequence")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    input_ids = torch.tensor(list(prompt_ids), dtype=torch.long, device=device)
    generated = generate(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        generator=generator,
    )
    token_ids = generated.detach().cpu().tolist()
    return tokenizer.decode(token_ids), token_ids


# A familiar alias for callers that use the term "decode" for generation.
decode = generate
