from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.generation import generate_token_ids
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text from an Assignment 1 language-model checkpoint."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional text file; an existing file is never overwritten.",
    )
    return parser.parse_args()


def require_files(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")


def main() -> None:
    args = parse_args()
    require_files(
        [
            args.config_path,
            args.checkpoint_path,
            args.vocab_path,
            args.merges_path,
        ]
    )
    if args.output_path is not None and args.output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA.")

    with args.config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    required_model_keys = (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "rope_theta",
    )
    missing_keys = [key for key in required_model_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Training config is missing model keys: {missing_keys}")

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=torch.float32,
    )
    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.vocab_path),
        merges_filepath=str(args.merges_path),
        special_tokens=["<|endoftext|>"],
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    prompt_token_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.special_token_to_id["<|endoftext|>"]
    generated_token_ids = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generated_text = tokenizer.decode(generated_token_ids)
    generated_token_count = len(generated_token_ids) - len(prompt_token_ids)
    stopped_by_eos = generated_token_count < args.max_new_tokens

    checkpoint_iteration = checkpoint.get("iteration", "unknown")
    print(f"checkpoint iteration: {checkpoint_iteration}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"prompt tokens: {len(prompt_token_ids)}")
    print(f"generated tokens: {generated_token_count}")
    print(f"stopped by EOS: {stopped_by_eos}")
    print(generated_text)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("x", encoding="utf-8") as output_file:
            output_file.write(generated_text)
            output_file.write("\n")
        print(f"generated text saved: {args.output_path}")


if __name__ == "__main__":
    main()
