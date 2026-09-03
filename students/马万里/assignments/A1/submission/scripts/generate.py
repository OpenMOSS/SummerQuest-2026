import argparse
import pickle
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
):
    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.special_token_to_id.get("<|endoftext|>", 0)]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = input_ids.copy()
    model.eval()

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_tensor)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :] / temperature

            # top-p 过滤
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            # 移除累积概率超过 top_p 的 token
            mask = cumulative_probs > top_p
            # 保证至少保留一个 token
            mask[0] = False
            sorted_logits[mask] = float("-inf")

            probs = torch.softmax(sorted_logits, dim=-1)
            next_token = sorted_indices[torch.multinomial(probs, 1)].item()

            generated_ids.append(next_token)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], device=device)], dim=1
            )

            # 遇到特殊 token 停止
            if next_token == tokenizer.special_token_to_id.get("<|endoftext|>", -1):
                break

    return tokenizer.decode(generated_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000.0)

    args = parser.parse_args()

    # 加载 tokenizer
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges_path, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])

    # 加载模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
    )
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=args.device,
    )
    print(output)


if __name__ == "__main__":
    main()