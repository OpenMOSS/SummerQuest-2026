"""Autoregressive decoding with temperature and nucleus sampling."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """Sample one token from a temperature-scaled top-p distribution."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        keep = cumulative - sorted_probabilities < top_p
        sorted_probabilities = sorted_probabilities * keep
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_sorted_index = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted_index)
    return torch.multinomial(probabilities, num_samples=1)


@torch.inference_mode()
def generate(
    model: TransformerLM,
    input_ids: list[int] | torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Append sampled tokens until EOS or ``max_new_tokens`` is reached."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    device = next(model.parameters()).device
    tokens = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] == 0:
        raise ValueError("input_ids must contain one non-empty prompt sequence")

    model.eval()
    for _ in range(max_new_tokens):
        model_input = tokens[:, -model.context_length :]
        next_logits = model(model_input)[:, -1, :]
        next_token = sample_next_token(next_logits, temperature=temperature, top_p=top_p)
        tokens = torch.cat((tokens, next_token), dim=-1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break
    return tokens.squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TransformerLM(**checkpoint["model_config"], device=device)
    model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    prompt_ids = tokenizer.encode(args.prompt)
    eos_bytes = args.special_token[0].encode("utf-8") if args.special_token else None
    eos_id = tokenizer.inverse_vocab.get(eos_bytes) if eos_bytes is not None else None
    output_ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        eos_token_id=eos_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(tokenizer.decode(output_ids.tolist()))


if __name__ == "__main__":
    main()
