"""A1 Transformer 语言模型生成命令入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from cs336_basics.generate import generate_token_ids, load_model, resolve_device
from cs336_basics.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    """解析文本生成参数。"""
    parser = argparse.ArgumentParser(description="使用训练好的 CS336 Transformer LM 生成文本")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="训练 config.json；默认读取 checkpoint 同目录下的 config.json",
    )
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--special-token", default="<|endoftext|>")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, help="可选的 UTF-8 文本输出文件")
    return parser.parse_args()


def main() -> None:
    """加载 tokenizer 和模型，生成续写并输出文本。"""
    args = parse_args()
    config_path = args.config or args.checkpoint.parent / "config.json"
    device = resolve_device(args.device)

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=str(args.vocab),
        merges_filepath=str(args.merges),
        special_tokens=[args.special_token],
    )
    prompt_ids = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.token_to_id[args.special_token.encode("utf-8")]
    model, iteration = load_model(
        checkpoint_path=args.checkpoint,
        config_path=config_path,
        device=device,
    )
    generated_ids = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        eos_token_id=eos_token_id,
    )
    generated_text = tokenizer.decode(generated_ids)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated_text, encoding="utf-8")

    iteration_text = "未知" if iteration is None else str(iteration)
    print(f"device={device} checkpoint_step={iteration_text} tokens={len(generated_ids)}")
    print(generated_text)


if __name__ == "__main__":
    main()
