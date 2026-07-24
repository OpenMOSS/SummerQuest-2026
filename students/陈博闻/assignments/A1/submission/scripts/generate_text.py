from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe import Tokenizer
from cs336_basics.generation import generate_text
from cs336_basics.transformer import TransformerLM


def _load_hex_vocab(path: Path) -> dict[int, bytes]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(token_id): bytes.fromhex(token_hex) for token_id, token_hex in raw.items()}


def _load_hex_merges(path: Path) -> list[tuple[bytes, bytes]]:
    merges: list[tuple[bytes, bytes]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            left, right = line.split()
            merges.append((bytes.fromhex(left), bytes.fromhex(right)))
    return merges


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpointed Transformer LM.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = Tokenizer(_load_hex_vocab(Path(args.vocab)), _load_hex_merges(Path(args.merges)), args.special_token)
    eos_token_id = tokenizer.bytes_to_id.get(b"<|endoftext|>")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
    )


if __name__ == "__main__":
    main()
