#!/usr/bin/env python3
"""Generate text from an A1 checkpoint with temperature and nucleus sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.model import TransformerLM  # noqa: E402
from cs336_basics.tokenizer import Tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    with resolve(args.config).open(encoding="utf-8") as file:
        config = json.load(file)
    tokenizer = Tokenizer.from_files(
        resolve(args.vocab),
        resolve(args.merges),
        special_tokens=["<|endoftext|>"],
    )
    model = TransformerLM(**config["model"]).to(device)
    checkpoint = torch.load(resolve(args.checkpoint), map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    encoded_prompt = tokenizer.encode(args.prompt)
    eos_bytes = b"<|endoftext|>"
    eos_id = next((token_id for token_id, token in tokenizer.vocab.items() if token == eos_bytes), None)
    if not encoded_prompt:
        if eos_id is None:
            raise ValueError("empty prompt requires an <|endoftext|> token")
        encoded_prompt = [eos_id]

    inputs = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    generated = model.generate(
        inputs,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
    )[0].tolist()
    text = tokenizer.decode(generated)
    result = {
        "prompt": args.prompt,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "requested_new_tokens": args.max_new_tokens,
        "generated_new_tokens": len(generated) - len(encoded_prompt),
        "stopped_on_eos": bool(generated and eos_id is not None and generated[-1] == eos_id),
        "text": text,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
