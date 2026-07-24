from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Assignment 1 checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--merges", required=True, type=Path)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    special_token = "<|endoftext|>"
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [special_token])
    device = torch.device(args.device)
    model = TransformerLM(**config["model"], device=device).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), device=device, dtype=torch.long)
    end_token_id = tokenizer.bytes_to_id[special_token.encode("utf-8")]
    output_ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        end_token_id=end_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    text = tokenizer.decode(output_ids.tolist())
    print(text)
    if args.output is not None:
        record = {
            "prompt": args.prompt,
            "text": text,
            "prompt_tokens": len(prompt_ids),
            "total_tokens": len(output_ids),
            "new_tokens": len(output_ids) - len(prompt_ids),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "stopped_on_eos": int(output_ids[-1]) == end_token_id,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
