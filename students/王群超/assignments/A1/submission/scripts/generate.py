from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.transformer import TransformerLM
from cs336_basics.bpe_tokenizer import BPETokenizer


def load_model(checkpoint_path: str, config: dict, device: str) -> TransformerLM:
    """从 checkpoint 加载模型权重。"""
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        device=torch.device(device),
        dtype=torch.float32,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # checkpoint 可能是 {model, optimizer, iteration} 或纯 state_dict
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p 过滤：保留累计概率 >= top_p 的最小 token 集合，其余设为 -inf。"""
    if top_p >= 1.0:
        return logits
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 保留累计概率 < top_p 的 token，以及刚好使累计概率 >= top_p 的那一个
    mask = cumsum - sorted_probs < top_p
    sorted_logits = torch.full_like(logits, float("-inf"))
    sorted_logits[sorted_idx[mask]] = logits[sorted_idx[mask]]
    return sorted_logits


def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
) -> str:
    """自回归生成文本。

    Args:
        model: 训练好的 TransformerLM
        tokenizer: 对应的 BPETokenizer
        prompt: 输入提示文本
        max_tokens: 最多生成多少个 token
        temperature: 采样温度，越低越确定
        top_p: nucleus sampling 的 p 值
        device: 计算设备

    Returns:
        生成的完整文本（prompt + 生成内容）
    """
    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    if len(input_ids) == 0:
        input_ids = [0]

    context_length = model.context_length
    eos_id = tokenizer._special_to_id.get("<|endoftext|>")

    generated = list(input_ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            # 截取最后 context_length 个 token
            context = generated[-context_length:]
            x = torch.tensor([context], dtype=torch.long, device=device)

            # 前向传播，取最后位置的 logits
            logits = model(x)
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)

            # Top-p 过滤
            if top_p < 1.0:
                next_logits = top_p_filter(next_logits, top_p)

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            # 遇到 EOS 停止
            if eos_id is not None and next_token == eos_id:
                break

    # 解码
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Transformer LM 文本生成")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint.pt 路径")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="tokenizer.pkl 路径")
    parser.add_argument("--prompt", type=str, default="",
                        help="输入提示文本")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cpu")

    # 模型架构参数（必须与训练时一致）
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    args = parser.parse_args()

    # 加载 tokenizer
    with open(args.tokenizer, "rb") as f:
        tok_data = pickle.load(f)
    tokenizer = BPETokenizer(
        vocab=tok_data["vocab"],
        merges=tok_data["merges"],
        special_tokens=tok_data.get("special_tokens", ["<|endoftext|>"]),
    )

    # 加载模型
    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
    }
    model = load_model(args.checkpoint, config, args.device)

    # 生成
    output = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )

    print("=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
