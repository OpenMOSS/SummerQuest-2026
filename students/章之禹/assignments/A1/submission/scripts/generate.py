"""Generate text from an A1 Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cs336_basics.generation import generate_text
from cs336_basics.training import build_model, load_config, load_model_checkpoint, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training/generation TOML config")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (overrides [generation].checkpoint)")
    parser.add_argument("--prompt", default=None, help="Prompt text (overrides [generation].prompt)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print decoded text and token IDs as JSON")
    return parser.parse_args()


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise TypeError(f"[{name}] must be a TOML table")
    return section


def _input_path(value: str, config: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(config["_config_path"]).parent
    for candidate in (Path.cwd() / path, config_dir / path, config_dir.parent / path):
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _load_tokenizer(config: dict[str, Any]):
    from cs336_basics.tokenizer import Tokenizer

    tokenizer_cfg = _section(config, "tokenizer")
    vocab = tokenizer_cfg.get("vocab_path", tokenizer_cfg.get("vocab"))
    merges = tokenizer_cfg.get("merges_path", tokenizer_cfg.get("merges"))
    if vocab is None or merges is None:
        raise KeyError("[tokenizer] must define vocab_path and merges_path")
    special_tokens = tokenizer_cfg.get("special_tokens", ["<|endoftext|>"])
    return Tokenizer.from_files(_input_path(str(vocab), config), _input_path(str(merges), config), special_tokens)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    generation_cfg = _section(config, "generation")
    training_cfg = _section(config, "training")
    device = resolve_device(training_cfg.get("device"))

    checkpoint_value = args.checkpoint or generation_cfg.get("checkpoint")
    if checkpoint_value is None:
        output_dir = _section(config, "run").get("output_dir", training_cfg.get("output_dir", "runs/default"))
        checkpoint_value = str(Path(str(output_dir)) / "latest.pt")
    checkpoint_path = _input_path(str(checkpoint_value), config)

    model = build_model(config, device)
    iteration, _ = load_model_checkpoint(checkpoint_path, model, map_location=device)
    tokenizer = _load_tokenizer(config)

    prompt = args.prompt if args.prompt is not None else generation_cfg.get("prompt")
    if prompt is None:
        raise ValueError("provide --prompt or generation.prompt")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(generation_cfg.get("max_new_tokens", 256))
    temperature = args.temperature if args.temperature is not None else float(generation_cfg.get("temperature", 1.0))
    top_p = args.top_p if args.top_p is not None else float(generation_cfg.get("top_p", 1.0))
    seed = args.seed if args.seed is not None else int(generation_cfg.get("seed", training_cfg.get("seed", 1337)))
    torch.manual_seed(seed)

    eos_text = generation_cfg.get("eos_token", "<|endoftext|>")
    eos_ids = tokenizer.encode(eos_text) if eos_text else []
    eos_token_id = int(eos_ids[0]) if len(eos_ids) == 1 else None
    text, token_ids = generate_text(
        model,
        tokenizer,
        str(prompt),
        max_new_tokens=max_new_tokens,
        context_length=int(_section(config, "model")["context_length"]),
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )
    if args.json:
        print(
            json.dumps(
                {"iteration": iteration, "seed": seed, "text": text, "token_ids": token_ids},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(text)


if __name__ == "__main__":
    main()
