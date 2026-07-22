from __future__ import annotations

import argparse
import json

import torch

from cs336_basics.generation import generate_text
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer_experiments import load_tokenizer_artifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer checkpoint.")
    parser.add_argument("--config", required=True, help="Training JSON containing the model section.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--eos-token", default="<|endoftext|>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with open(args.config, encoding="utf-8") as f:
        configuration = json.load(f)

    device = torch.device(args.device)
    model = TransformerLM(**configuration["model"], device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state)

    tokenizer = load_tokenizer_artifact(args.tokenizer)
    eos_token_id = None
    if args.eos_token:
        eos_bytes = args.eos_token.encode("utf-8")
        eos_token_id = next(
            (token_id for token_id, token_bytes in tokenizer.vocab.items() if token_bytes == eos_bytes),
            None,
        )
        if eos_token_id is None:
            raise ValueError(f"EOS token is not present in the tokenizer vocabulary: {args.eos_token!r}")
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        generator=generator,
    )
    print(result.text)


if __name__ == "__main__":
    main()
