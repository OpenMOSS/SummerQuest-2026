#!/usr/bin/env python3
"""Sample text from a trained checkpoint with temperature and top-p."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.experiment import build_model, load_json, set_seed
from cs336_basics.tokenizer import Tokenizer


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities[remove] = 0
    sorted_probabilities /= sorted_probabilities.sum()
    sampled_rank = torch.multinomial(sorted_probabilities, num_samples=1)
    return int(sorted_indices[sampled_rank].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_json(args.config)
    tokenizer = Tokenizer.load(args.tokenizer)
    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    token_ids = tokenizer.encode(args.prompt)
    eos_bytes = b"<|endoftext|>"
    eos_id = next((token_id for token_id, value in tokenizer.vocab.items() if value == eos_bytes), None)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = token_ids[-config["model"]["context_length"] :]
            inputs = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(inputs)[0, -1].float()
            next_id = sample_top_p(logits, args.temperature, args.top_p)
            token_ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
    text = tokenizer.decode(token_ids)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
