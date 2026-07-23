from __future__ import annotations

import torch

from cs336_basics.transformer import softmax


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest high-probability set whose mass is at least top_p."""

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    keep_sorted = cumulative_probs <= top_p
    keep_sorted[..., 0] = True

    first_over_p = torch.argmax((cumulative_probs >= top_p).to(torch.long), dim=-1)
    keep_sorted[first_over_p] = True

    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[sorted_indices[keep_sorted]] = True

    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    return filtered / torch.sum(filtered, dim=-1, keepdim=True)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> int:
    """Sample one token id from next-token logits."""

    if temperature < 0:
        raise ValueError("temperature must be non-negative.")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probs = softmax(logits / temperature, dim=-1)
    if top_p is not None:
        probs = apply_top_p(probs, top_p)

    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())


@torch.no_grad()
def generate_token_ids(
    model: torch.nn.Module,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
) -> list[int]:
    """Generate token ids by repeatedly sampling from the model."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must contain at least one token.")

    was_training = model.training
    model.eval()

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    generated = list(prompt_token_ids)
    context_length = getattr(model, "context_length", None)

    for _ in range(max_new_tokens):
        input_ids = generated
        if context_length is not None:
            input_ids = input_ids[-context_length:]

        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(x)
        next_token_logits = logits[0, -1]
        next_token_id = sample_next_token(next_token_logits, temperature=temperature, top_p=top_p)

        generated.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    if was_training:
        model.train()

    return generated


def generate_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float | None = None,
    device: str | torch.device | None = None,
) -> str:
    """Encode a prompt, generate token ids, and decode them back to text."""

    prompt_ids = tokenizer.encode(prompt)
    generated_ids = generate_token_ids(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )
    return tokenizer.decode(generated_ids)
