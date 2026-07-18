import torch

from cs336_basics.softmax import softmax

"""
生成文本
"""

def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Lower temperature makes the distribution sharper (more deterministic).
    Higher temperature makes it flatter (more random).
    """
    return logits / temperature


def top_p_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling.

    Sorts probabilities, takes the smallest set of tokens whose cumulative
    probability >= p, renormalizes, and samples.
    """
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Mask tokens beyond the threshold p
    mask = cumulative > p
    # Shift mask so the first token that reaches p is included
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from the filtered distribution
    sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(dim=-1, index=sampled_idx).squeeze(-1)


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int = None,
) -> list[int]:
    """Generate text from a language model.

    Args:
        model: TransformerLM instance.
        prompt: (1, prompt_len) tensor of token ids.
        max_tokens: Maximum number of new tokens to generate.
        temperature: Softmax temperature (> 0, lower = more deterministic).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        end_token_id: Token id that signals end-of-generation.

    Returns:
        List of generated token ids (prompt + new tokens).
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    generated = prompt[0].tolist()
    context_length = model.context_length

    for _ in range(max_tokens):
        # Truncate to context window if needed
        if len(generated) > context_length:
            input_ids = torch.tensor([generated[-context_length:]], device=device)
        else:
            input_ids = torch.tensor([generated], device=device)

        # Forward pass: (1, seq_len, vocab_size)
        logits = model(input_ids)
        # Take the last position's prediction
        next_logits = logits[0, -1, :]

        # Temperature scaling
        if temperature != 1.0:
            next_logits = temperature_scale(next_logits, temperature)

        # Softmax → probabilities
        probs = softmax(next_logits, dim=-1)

        # Top-p sampling
        if top_p < 1.0:
            next_token = top_p_sample(probs, top_p)
        else:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        token_id = next_token.item()
        generated.append(token_id)

        if end_token_id is not None and token_id == end_token_id:
            break

    return generated