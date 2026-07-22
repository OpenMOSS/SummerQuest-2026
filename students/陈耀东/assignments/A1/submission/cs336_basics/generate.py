"""Transformer 语言模型的自回归文本生成工具。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor

from cs336_basics.nn_modules import TransformerLM


MODEL_CONFIG_KEYS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "rope_theta",
)


def resolve_device(requested_device: str) -> torch.device:
    """解析推理设备；auto 在 CUDA 可用时优先选择 CUDA。"""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """从一维 logits 中采样下一个 token。

    ``temperature=0`` 使用贪心解码。否则先缩放 logits，再只保留累计概率
    达到 ``top_p`` 所需的最小候选集合，最后从重新归一化后的分布中采样。
    """
    if logits.ndim != 1:
        raise ValueError("logits 必须是一维 vocab 向量")
    if temperature < 0:
        raise ValueError("temperature 不能小于 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1] 区间")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        # 保留第一个使累计概率达到 top_p 的 token，保证候选集合永不为空。
        remove_mask = cumulative_probabilities - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove_mask, 0.0)
        sorted_probabilities = sorted_probabilities / sorted_probabilities.sum()

        sampled_rank = torch.multinomial(sorted_probabilities, 1, generator=generator)
        return int(sorted_indices[sampled_rank].item())

    sampled_token = torch.multinomial(probabilities, 1, generator=generator)
    return int(sampled_token.item())


@torch.inference_mode()
def generate_token_ids(
    model: TransformerLM,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    seed: int = 42,
    eos_token_id: int | None = None,
) -> list[int]:
    """根据 prompt 自回归生成 token，并在采到 EOS 时停止。

    返回值包含 prompt 和生成的普通 token，但不包含触发停止的 EOS。每一步仅使用
    最后 ``model.context_length`` 个 token 作为输入，因此长 prompt 和长续写不会超过
    模型训练时的上下文窗口。
    """
    if not prompt_ids:
        raise ValueError("prompt 编码后不能为空")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能小于 0")

    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    all_token_ids = [int(token_id) for token_id in prompt_ids]
    token_tensor = torch.tensor(all_token_ids, dtype=torch.long, device=device).unsqueeze(0)
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            model_input = token_tensor[:, -model.context_length :]
            next_token_logits = model(model_input)[0, -1]
            next_token_id = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            all_token_ids.append(next_token_id)
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            token_tensor = torch.cat((token_tensor, next_token), dim=1)
    finally:
        model.train(was_training)

    return all_token_ids


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
) -> tuple[TransformerLM, int | None]:
    """从训练配置和 checkpoint 恢复只用于推理的模型。"""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    missing_keys = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing_keys:
        missing_text = ", ".join(missing_keys)
        raise KeyError(f"训练配置缺少模型字段：{missing_text}")

    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        device=device,
        normalization=config.get("normalization", "pre"),
        positional_encoding=config.get("positional_encoding", "rope"),
        ffn_type=config.get("ffn_type", "swiglu"),
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint 必须包含 model state_dict")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    iteration = checkpoint.get("iteration")
    return model, None if iteration is None else int(iteration)
