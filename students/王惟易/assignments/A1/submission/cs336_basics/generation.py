import torch

from cs336_basics.model import softmax

def sample_next_token(
        logits: torch.Tensor,       # (vocab_size, )
        temperature: float=1.0,
        top_p: float=1.0,
) -> torch.Tensor:                  # (1,), dtype=torch.long
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 标记需要移除的 token
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = False # 至少保留第一个

    if top_p < 1.0:
        sorted_probs = sorted_probs.masked_fill(
            sorted_indices_to_remove, 0.0
        )
        sorted_probs = sorted_probs / sorted_probs.sum()

    sampled_position =  torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[sampled_position]

    return next_token

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        context = generated[-context_length:]       # (T)
        context = context.unsqueeze(0)              # (1, T)
        logits = model(context)                     # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :]               # (vocab_size)
        next_token = sample_next_token(next_token_logits, temperature=temperature, top_p=top_p)
        generated = torch.cat([generated, next_token])
        if (
            eos_token_id is not None and next_token.item() == eos_token_id
        ):
            break

    model.train(was_training)
    return generated