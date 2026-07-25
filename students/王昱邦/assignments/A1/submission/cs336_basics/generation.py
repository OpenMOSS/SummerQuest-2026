from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from cs336_basics.model import softmax
from cs336_basics.tokenizer import Tokenizer


def _validate_top_p(top_p: float) -> None:
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1].")


def apply_top_p(probabilities: Tensor, top_p: float) -> Tensor:
    """Keep the smallest high-probability prefix whose mass reaches top_p."""
    _validate_top_p(top_p)
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be one-dimensional.")
    if probabilities.numel() == 0:
        raise ValueError("probabilities must not be empty.")

    working_probabilities = probabilities.to(dtype=torch.float32)
    if not torch.isfinite(working_probabilities).all():
        raise ValueError("probabilities must be finite.")
    if (working_probabilities < 0).any():
        raise ValueError("probabilities must be non-negative.")

    total_probability = working_probabilities.sum()
    if total_probability <= 0:
        raise ValueError("probabilities must have positive total mass.")
    normalized_probabilities = working_probabilities / total_probability

    sorted_probabilities, sorted_indices = torch.sort(
        normalized_probabilities,
        descending=True,
    )
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=0)

    # Keep every item whose preceding cumulative mass is below top_p. This
    # includes the first item that makes the cumulative mass reach top_p.
    keep_sorted = cumulative_probabilities - sorted_probabilities < top_p
    filtered_sorted = torch.where(
        keep_sorted,
        sorted_probabilities,
        torch.zeros_like(sorted_probabilities),
    )
    filtered_probabilities = torch.zeros_like(normalized_probabilities)
    filtered_probabilities.scatter_(
        dim=0,
        index=sorted_indices,
        src=filtered_sorted,
    )
    return filtered_probabilities / filtered_probabilities.sum()


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token ID from a one-dimensional next-token logit vector."""
    _validate_top_p(top_p)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional.")
    if logits.numel() == 0:
        raise ValueError("logits must not be empty.")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite.")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits.to(dtype=torch.float32) / temperature
    probabilities = softmax(scaled_logits, dim=-1)
    filtered_probabilities = apply_top_p(probabilities, top_p=top_p)
    sampled_token = torch.multinomial(
        filtered_probabilities,
        num_samples=1,
        generator=generator,
    )
    return int(sampled_token.item())


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    eos_token_id: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively extend prompt_token_ids, excluding a sampled EOS."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if eos_token_id < 0:
        raise ValueError("eos_token_id must be non-negative.")

    generated_token_ids = [int(token_id) for token_id in prompt_token_ids]
    if not generated_token_ids:
        raise ValueError("prompt_token_ids must contain at least one token.")
    if any(token_id < 0 for token_id in generated_token_ids):
        raise ValueError("prompt token IDs must be non-negative.")

    context_length = getattr(model, "context_length", None)
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("model must expose a positive integer context_length.")

    vocab_size = getattr(model, "vocab_size", None)
    if isinstance(vocab_size, int):
        if eos_token_id >= vocab_size:
            raise ValueError("eos_token_id is outside the model vocabulary.")
        if any(token_id >= vocab_size for token_id in generated_token_ids):
            raise ValueError("prompt contains a token outside the model vocabulary.")

    if generated_token_ids[-1] == eos_token_id or max_new_tokens == 0:
        return generated_token_ids

    device = _model_device(model)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                context_token_ids = generated_token_ids[-context_length:]
                model_input = torch.tensor(
                    [context_token_ids],
                    dtype=torch.long,
                    device=device,
                )
                logits = model(model_input)
                if logits.ndim != 3 or logits.shape[0] != 1:
                    raise ValueError(
                        "model must return logits with shape "
                        "(1, sequence_length, vocab_size)."
                    )

                next_token_id = sample_next_token(
                    logits=logits[0, -1, :],
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )
                if next_token_id == eos_token_id:
                    break
                generated_token_ids.append(next_token_id)
    finally:
        model.train(was_training)

    return generated_token_ids


def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_of_text_token: str = "<|endoftext|>",
    generator: torch.Generator | None = None,
) -> str:
    """Generate a text completion and return the prompt plus completion."""
    try:
        eos_token_id = tokenizer.special_token_to_id[end_of_text_token]
    except KeyError as exc:
        raise ValueError(
            f"Tokenizer does not contain special token {end_of_text_token!r}."
        ) from exc

    prompt_token_ids = tokenizer.encode(prompt)
    generated_token_ids = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        generator=generator,
    )
    return tokenizer.decode(generated_token_ids)
