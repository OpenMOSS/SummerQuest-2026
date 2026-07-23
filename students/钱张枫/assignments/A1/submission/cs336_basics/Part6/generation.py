from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import torch
from torch import nn


class TokenizerProtocol(Protocol):
    """生成模块所需的最小 tokenizer 接口。"""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


def generate_token_ids(
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """使用自回归 nucleus sampling 生成 token ID。

    返回值包含原始 prompt。每一步只使用模型最后一个序列位置的 logits；当
    prompt 与已生成内容超过模型的 ``context_length`` 时，仅保留最近的上下文
    作为下一步模型输入，但返回值仍保留完整序列。

    Args:
        model: 返回 ``(batch, sequence, vocab)`` logits 且具有正整数
            ``context_length`` 属性的 PyTorch 模型。
        prompt_token_ids: 非空 prompt token ID 序列。
        max_new_tokens: 最多新增的 token 数，允许为零。
        temperature: 正的 temperature scaling 参数。
        top_p: nucleus sampling 的累计概率阈值，取值范围为 ``(0, 1]``。
        eot_token_id: 可选的 end-of-text token ID。采样到该 ID 后立即停止。
        generator: 可选的 ``torch.Generator``，用于可复现采样。

    Raises:
        TypeError: 参数类型不合法。
        ValueError: 参数范围、模型输出或 logits 不合法。
    """

    context_length = _validate_generation_arguments(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eot_token_id=eot_token_id,
        generator=generator,
    )
    generated_token_ids = _normalize_token_ids(prompt_token_ids)
    if max_new_tokens == 0:
        return generated_token_ids

    device = _get_model_device(model)
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                model_input_ids = generated_token_ids[-context_length:]
                model_input = torch.tensor(model_input_ids, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(model_input)
                next_token_logits = _get_last_position_logits(logits, expected_sequence_length=len(model_input_ids))
                next_token_id = _sample_next_token(
                    next_token_logits,
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )
                generated_token_ids.append(next_token_id)

                if eot_token_id is not None and next_token_id == eot_token_id:
                    break
    finally:
        model.train(was_training)

    return generated_token_ids


def generate_text(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eot_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """编码 prompt、生成 token，并解码包含 prompt 的完整文本。"""

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    prompt_token_ids = tokenizer.encode(prompt)
    generated_token_ids = generate_token_ids(
        model,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eot_token_id=eot_token_id,
        generator=generator,
    )
    return tokenizer.decode(generated_token_ids)


def _validate_generation_arguments(
    *,
    model: nn.Module,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eot_token_id: int | None,
    generator: torch.Generator | None,
) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")

    context_length = getattr(model, "context_length", None)
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("model.context_length must be an integer.")
    if context_length <= 0:
        raise ValueError("model.context_length must be greater than zero.")

    if isinstance(prompt_token_ids, (str, bytes)) or not isinstance(prompt_token_ids, Sequence):
        raise TypeError("prompt_token_ids must be a sequence of integers.")
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must not be empty.")
    _normalize_token_ids(prompt_token_ids)

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must not be negative.")

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a real number.")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero.")

    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise TypeError("top_p must be a real number.")
    if not math.isfinite(float(top_p)) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in the interval (0, 1].")

    if eot_token_id is not None:
        _validate_token_id(eot_token_id, argument_name="eot_token_id")

    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an instance of torch.Generator or None.")

    return context_length


def _normalize_token_ids(token_ids: Sequence[int]) -> list[int]:
    normalized_token_ids: list[int] = []
    for index, token_id in enumerate(token_ids):
        _validate_token_id(token_id, argument_name=f"prompt_token_ids[{index}]")
        normalized_token_ids.append(token_id)
    return normalized_token_ids


def _validate_token_id(token_id: int, *, argument_name: str) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError(f"{argument_name} must be an integer.")
    if token_id < 0:
        raise ValueError(f"{argument_name} must not be negative.")


def _get_model_device(model: nn.Module) -> torch.device:
    first_parameter = next(model.parameters(), None)
    if first_parameter is not None:
        return first_parameter.device

    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return first_buffer.device

    return torch.device("cpu")


def _get_last_position_logits(logits: object, *, expected_sequence_length: int) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model output must be a torch.Tensor.")
    if logits.ndim != 3:
        raise ValueError("model output must have shape (batch_size, sequence_length, vocab_size).")
    if logits.shape[0] != 1:
        raise ValueError("model output batch size must be one during generation.")
    if logits.shape[1] != expected_sequence_length:
        raise ValueError("model output sequence length must match the model input sequence length.")
    if logits.shape[2] <= 0:
        raise ValueError("model output vocabulary dimension must be greater than zero.")

    last_position_logits = logits[0, -1]
    if not torch.is_floating_point(last_position_logits):
        raise TypeError("model logits must have a floating-point dtype.")
    if not bool(torch.isfinite(last_position_logits).all().item()):
        raise ValueError("model logits must contain only finite values.")
    return last_position_logits


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    # float32 避免低精度 logits 在 temperature scaling 和累计求和时放大误差。
    scaled_logits = logits.to(dtype=torch.float32) / float(temperature)
    probabilities = torch.softmax(scaled_logits, dim=-1)
    sorted_probabilities, sorted_token_ids = torch.sort(probabilities, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    threshold = cumulative_probabilities.new_tensor(float(top_p))
    last_candidate_index = int(torch.searchsorted(cumulative_probabilities, threshold, right=False).item())
    candidate_count = min(last_candidate_index + 1, sorted_probabilities.numel())
    candidate_probabilities = sorted_probabilities[:candidate_count]
    candidate_token_ids = sorted_token_ids[:candidate_count]

    probability_mass = candidate_probabilities.sum()
    if not bool(torch.isfinite(probability_mass).item()) or probability_mass.item() <= 0:
        raise ValueError("nucleus sampling produced an invalid probability distribution.")

    normalized_probabilities = candidate_probabilities / probability_mass
    sampled_candidate_index = torch.multinomial(normalized_probabilities, 1, generator=generator)
    return int(candidate_token_ids[sampled_candidate_index].item())
