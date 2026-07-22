"""Generate text from a checkpoint produced by scripts/train_lm.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.bpe import BPETokenizer
from cs336_basics.decoding import generate
from cs336_basics.model import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10_000.0),
        device=args.device,
        norm_mode=config.get("norm_mode", "pre"),
        use_rope=config.get("use_rope", True),
        ffn_type=config.get("ffn_type", "swiglu"),
        tied_embeddings=config.get("tied_embeddings", False),
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    tokenizer = BPETokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_of_text_id=tokenizer.special_token_ids["<|endoftext|>"],
        device=args.device,
        generator=generator,
    )
    print(text)


if __name__ == "__main__":
    main()
