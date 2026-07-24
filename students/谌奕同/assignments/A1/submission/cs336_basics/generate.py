"""Generate text from a trained Transformer LM."""

import argparse
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling and softmax."""
    return softmax(logits / temperature, dim=-1)


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose probabilities sum to >= top_p."""
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Remove tokens beyond the nucleus.
    remove_mask = cumulative_probs > top_p
    # Keep at least the most probable token.
    remove_mask[..., 0] = False
    sorted_probs[remove_mask] = 0.0
    # Re-normalize.
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # Scatter back to original order.
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_indices, sorted_probs)
    return filtered


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: torch.device = torch.device("cpu"),
    eos_token: str = "<|endoftext|>",
) -> str:
    """Generate a completion for the given prompt."""
    model.eval()
    eos_token_id = tokenizer.special_token_ids.get(eos_token)
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = list(input_ids)
    for _ in range(max_new_tokens):
        # Truncate to context length if needed.
        if input_tensor.size(1) > model.context_length:
            input_tensor = input_tensor[:, -model.context_length :]

        logits = model(input_tensor)
        next_logits = logits[:, -1, :]  # (1, vocab_size)

        probs = softmax_with_temperature(next_logits, temperature)
        probs = top_p_filter(probs, top_p)

        # Sample from the filtered distribution.
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        next_token_id = next_token.item()
        generated_ids.append(next_token_id)

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer.from_files(
        args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"]
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    model = TransformerLM(
        vocab_size=config.get("vocab_size", 10_000),
        context_length=config.get("context_length", 256),
        d_model=config.get("d_model", 512),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 16),
        d_ff=config.get("d_ff", 1344),
        rope_theta=config.get("rope_theta", 10_000.0),
        use_rmsnorm=config.get("use_rmsnorm", True),
        use_post_norm=config.get("use_post_norm", False),
        use_rope=config.get("use_rope", True),
        ffn_type=config.get("ffn_type", "swiglu"),
        qk_norm=config.get("qk_norm", False),
        zero_init_output=config.get("zero_init_output", False),
    ).to(device)

    state_dict = checkpoint["model_state_dict"]
    # Strip the torch.compile _orig_mod. prefix if present.
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
