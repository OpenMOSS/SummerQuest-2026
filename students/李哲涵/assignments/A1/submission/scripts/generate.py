from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _project_api import (
    build_model,
    build_optimizer,
    get_load_checkpoint_fn,
    get_tokenizer_cls,
    load_config_for_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Transformer LM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path, required=True)
    parser.add_argument("--merges-path", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()
        sampled_sorted_index = torch.multinomial(sorted_probabilities, num_samples=1)
        return int(sorted_indices[sampled_sorted_index].item())

    return int(torch.multinomial(probabilities, num_samples=1).item())


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.vocab_path, args.merges_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens 不能为负数")
    if args.temperature < 0:
        raise ValueError("--temperature 不能为负数；0 表示 greedy decoding")
    if not (0.0 < args.top_p <= 1.0):
        raise ValueError("--top-p 必须位于 (0, 1]")

    torch.manual_seed(args.seed)
    config = load_config_for_checkpoint(args.checkpoint)
    config["device"] = args.device
    device = torch.device(args.device)

    with args.vocab_path.open("r", encoding="utf-8") as file:
        serialized_vocab = json.load(file)
    valid_token_ids = sorted(int(token_id) for token_id in serialized_vocab)
    if not valid_token_ids:
        raise RuntimeError("vocab 文件为空")
    if valid_token_ids[-1] >= int(config["vocab_size"]):
        raise ValueError(
            f"vocab 最大 token id={valid_token_ids[-1]} 超出模型 vocab_size={config['vocab_size']}"
        )
    valid_id_mask = torch.zeros(int(config["vocab_size"]), dtype=torch.bool, device=device)
    valid_id_mask[torch.tensor(valid_token_ids, dtype=torch.long, device=device)] = True

    tokenizer_cls = get_tokenizer_cls()
    tokenizer = tokenizer_cls.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        list(args.special_token),
    )

    model = build_model(config).to(device)
    # 使用训练时同构的 optimizer，让单测通过的 load_checkpoint 恢复状态。
    optimizer = build_optimizer(model, config)
    get_load_checkpoint_fn()(args.checkpoint, model, optimizer)
    model.eval()

    token_ids = list(tokenizer.encode(args.prompt))
    if not token_ids:
        raise ValueError("prompt 编码后为空；请提供非空 prompt")

    stop_ids: set[int] = set()
    for token in args.special_token:
        encoded = list(tokenizer.encode(token))
        if len(encoded) == 1:
            stop_ids.add(int(encoded[0]))

    context_length = int(config["context_length"])
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = token_ids[-context_length:]
            inputs = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(inputs)[0, -1]
            logits = logits.masked_fill(~valid_id_mask, float("-inf"))
            next_id = sample_top_p(logits, args.temperature, args.top_p)
            token_ids.append(next_id)
            if next_id in stop_ids:
                break

    text = tokenizer.decode(token_ids)
    print(text)
    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")
        print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
