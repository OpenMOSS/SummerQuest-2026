from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.artifacts import load_tokenizer
from cs336_basics.model import softmax
from scripts.train import build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Transformer checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probabilities = softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
    sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()
    sampled_sorted_index = torch.multinomial(sorted_probabilities, num_samples=1)
    return int(sorted_indices[sampled_sorted_index].item())


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = resolve_device(args.device or config["training"].get("device", "auto"))
    tokenizer = load_tokenizer(args.tokenizer)
    model = build_model(config["model"], device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    token_ids = tokenizer.encode(args.prompt)
    end_token_id = tokenizer.special_to_id.get("<|endoftext|>")
    for _ in range(args.max_new_tokens):
        context = token_ids[-model.context_length :]
        inputs = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        next_token = sample_next_token(model(inputs)[0, -1], args.temperature, args.top_p)
        token_ids.append(next_token)
        if end_token_id is not None and next_token == end_token_id:
            break
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
