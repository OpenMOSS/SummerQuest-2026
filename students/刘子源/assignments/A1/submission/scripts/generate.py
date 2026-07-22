# 命令行生成入口，参数多归多，最后就是把 prompt 喂进去续写。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.generation import generate  # noqa: E402
from cs336_basics.tokenizer import Tokenizer  # noqa: E402
from cs336_basics.training import build_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # seed 不钉死的话，每次跑出来都不一样，复现实验会很烦。
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, [args.special_token])
    model = build_model(config, device)
    save_data = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(save_data["model_state_dict"])
    model.eval()
    eos_id = tokenizer.bytes_to_id.get(args.special_token.encode("utf-8"))
    all_ids = generate(
        model,
        tokenizer.encode(args.prompt),
        eos_id,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        config["context_length"],
    )
    print(tokenizer.decode(all_ids))


if __name__ == "__main__":
    main()
