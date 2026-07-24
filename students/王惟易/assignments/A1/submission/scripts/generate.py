import torch
import json
import argparse
from pathlib import Path

from cs336_basics.generation import generate
from cs336_basics.bpe_io import load_bpe
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Let a decoder-only Transformer language model generate"
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    return parser.parse_args(argv)

def load_config(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    device = config["device"]
    seed = (
        config["seed"] if args.seed is None else args.seed
    )

    vocab, merges = load_bpe(args.tokenizer)
    tokenizer = Tokenizer(
        vocab,
        merges,
        special_tokens=[END_OF_TEXT]
    )

    model = TransformerLM(
        **config["model"],
        device=device
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model"])
    # 放到模型 load_state_dict 之后，避免随机数被随机模型初始化消耗
    torch.manual_seed(seed)

    input_ids = torch.tensor(
        tokenizer.encode(args.prompt),
        dtype=torch.long,
        device=device,
    )

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=config["model"]["context_length"],
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.special_token_to_id[END_OF_TEXT],
    )

    text = tokenizer.decode(output_ids.tolist())
    print(text)

if __name__ == "__main__":
    main()
