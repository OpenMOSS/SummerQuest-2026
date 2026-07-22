#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def top_p_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probabilities = softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities[remove] = 0
    sorted_probabilities /= sorted_probabilities.sum()
    sampled_rank = torch.multinomial(sorted_probabilities, 1)
    return int(sorted_indices[sampled_rank])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = json.load(file)
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    model = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config.get("rope_theta", 10000.0),
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.manual_seed(args.seed)
    ids = tokenizer.encode(args.prompt)
    end_id = tokenizer.special_to_id.get("<|endoftext|>")
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = torch.tensor([ids[-config["context_length"] :]], dtype=torch.long, device=args.device)
            next_id = top_p_sample(model(context)[0, -1], args.temperature, args.top_p)
            ids.append(next_id)
            if next_id == end_id:
                break
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
