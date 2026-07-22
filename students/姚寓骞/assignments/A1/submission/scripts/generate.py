from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature must be positive and top-p must lie in (0, 1]")

    config = json.loads(args.config.read_text())
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    with args.tokenizer.open("rb") as file:
        tokenizer = Tokenizer(**pickle.load(file))
    model = TransformerLM(**config["model"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    model.eval()
    ids = tokenizer.encode(args.prompt)
    stop_id = tokenizer.special_to_id.get("<|endoftext|>")
    if not ids:
        if stop_id is None:
            raise ValueError("empty prompt requires a tokenizer with an <|endoftext|> token")
        # Treat the document-boundary token as BOS for unconditional generation.
        ids = [stop_id]

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
            logits = model(context)[0, -1] / args.temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probabilities = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probabilities, dim=-1)
            remove = cumulative - probabilities >= args.top_p
            sorted_logits[remove] = -torch.inf
            sampled = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
            token_id = sorted_indices[sampled].item()
            ids.append(token_id)
            if stop_id is not None and token_id == stop_id:
                break
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
