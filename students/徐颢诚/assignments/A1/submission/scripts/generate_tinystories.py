"""从 TinyStories checkpoint 生成文本，并保存可提交的文本 dump。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cs336_basics.decoding import generate
from cs336_basics.nn import TransformerLM
from cs336_basics.tokenizer import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRECTORY = ROOT / "logs" / "ts_baseline" / "2026-07-15" / "08-16-44"
SPECIAL_TOKEN = "<|endoftext|>"


def resolve_device(device_name: str) -> torch.device:
    """解析推理设备，默认优先 CUDA。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_tokenizer(path: Path) -> Tokenizer:
    """加载 BPE 训练阶段序列化的词表和 merge 规则。"""
    with path.open("rb") as file:
        serialized = pickle.load(file)
    return Tokenizer(serialized["vocab"], serialized["merges"], serialized.get("special_tokens", [SPECIAL_TOKEN]))


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device) -> TransformerLM:
    """按训练配置重建模型，并只恢复 checkpoint 中的模型参数。"""
    config = OmegaConf.load(config_path)
    dtype = getattr(torch, str(config.model.dtype))
    model = TransformerLM(
        config.data.vocab_size,
        config.model.context_length,
        config.model.d_model,
        config.model.num_layers,
        config.model.num_heads,
        config.model.d_ff,
        config.model.rope_theta,
        device=device,
        dtype=dtype,
        use_rmsnorm=bool(config.model.get("use_rmsnorm", True)),
        use_rope=bool(config.model.get("use_rope", True)),
        use_swiglu=bool(config.model.get("use_swiglu", True)),
        norm_position=str(config.model.get("norm_position", "pre")),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIRECTORY / "latest.pt")
    parser.add_argument("--config", type=Path, default=DEFAULT_RUN_DIRECTORY / "config.yaml")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts" / "tinystories_bpe_10k.pkl")
    parser.add_argument("--prompt", default=f"{SPECIAL_TOKEN}Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN_DIRECTORY / "generation.txt")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")

    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.checkpoint, args.config, device)
    eos_token_id = tokenizer.special_token_ids[SPECIAL_TOKEN]
    prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device)
    torch.manual_seed(args.seed)
    generated_ids = generate(
        model,
        prompt_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    generated_text = tokenizer.decode(generated_ids.tolist())
    new_token_count = generated_ids.numel() - prompt_ids.numel()
    stopped_at_eos = generated_ids[-1].item() == eos_token_id
    metadata = (
        f"# checkpoint: {args.checkpoint}\n"
        f"# prompt: {args.prompt!r}\n"
        f"# temperature: {args.temperature}\n"
        f"# top_p: {args.top_p}\n"
        f"# seed: {args.seed}\n"
        f"# generated_new_tokens: {new_token_count}\n"
        f"# stopped_at_eos: {stopped_at_eos}\n\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(metadata + generated_text + "\n", encoding="utf-8")
    print(metadata + generated_text)
    print(f"\n文本 dump 已写入 {args.output}")


if __name__ == "__main__":
    main()
