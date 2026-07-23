from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.Part2.tokenizer import Tokenizer
from cs336_basics.Part5.configuration import load_experiment_config
from cs336_basics.Part5.experiment_logging import write_json_atomic
from cs336_basics.Part5.training import load_model_for_inference, resolve_project_path
from cs336_basics.Part6.generation import generate_token_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 Transformer checkpoint.")
    parser.add_argument("--config", required=True, help="Training config used to construct the model.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by scripts/train_lm.py.")
    parser.add_argument("--prompt", default="Once upon a time", help="Prompt text. Must encode to at least one token.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Override the device from the training config.")
    parser.add_argument("--output", help="Optional UTF-8 text output path.")
    parser.add_argument("--metadata-output", help="Optional JSON metadata path for reproducibility.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seed < 0:
        raise ValueError("--seed must not be negative.")

    config = load_experiment_config(resolve_project_path(args.config))
    model, device = load_model_for_inference(config, args.checkpoint, device_override=args.device)
    tokenizer = Tokenizer.from_files(
        str(resolve_project_path(config.tokenizer.vocab_path)),
        str(resolve_project_path(config.tokenizer.merges_path)),
        list(config.tokenizer.special_tokens),
    )

    torch.manual_seed(args.seed)
    generator: torch.Generator | None = None
    if device.type in ("cpu", "cuda"):
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)

    prompt_token_ids = tokenizer.encode(args.prompt)
    generated_token_ids = generate_token_ids(
        model,
        prompt_token_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eot_token_id=config.tokenizer.eot_token_id,
        generator=generator,
    )
    text = tokenizer.decode(generated_token_ids)
    generated_count = len(generated_token_ids) - len(prompt_token_ids)
    stopped_on_eot = (
        config.tokenizer.eot_token_id is not None
        and generated_count > 0
        and generated_token_ids[-1] == config.tokenizer.eot_token_id
    )
    if args.output:
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    metadata_output = args.metadata_output
    if metadata_output is None and args.output:
        requested_output_path = Path(args.output)
        metadata_output = str(requested_output_path.with_suffix(requested_output_path.suffix + ".json"))
    if metadata_output:
        write_json_atomic(
            resolve_project_path(metadata_output),
            {
                "experiment_name": config.experiment_name,
                "config": args.config,
                "checkpoint": args.checkpoint,
                "prompt": args.prompt,
                "prompt_token_count": len(prompt_token_ids),
                "generated_token_count": generated_count,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "eot_token_id": config.tokenizer.eot_token_id,
                "stopped_on_eot": stopped_on_eot,
            },
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
