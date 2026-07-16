from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer_io import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model_config = json.loads(Path(args.config).read_text(encoding="utf-8"))["model"]
    model = TransformerLM(**model_config, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    end_token_id = tokenizer.special_token_id("<|endoftext|>")
    output_ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token_id=end_token_id,
        device=device,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
