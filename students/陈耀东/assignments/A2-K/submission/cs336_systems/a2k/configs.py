"""A2 讲义中的模型规模与实验默认值。

把配置集中在一个文件中有两个目的：

1. benchmark、显存实验和 Nsight profile 使用完全相同的模型定义；
2. 结构化结果只要记录 ``model_size`` 和展开后的字段，就能复现实验，避免
   不同脚本分别手抄超参数而产生悄无声息的偏差。

``tiny`` 不是讲义计分配置，只用于本地 CPU 冒烟测试。其余五个配置来自
Assignment 2 讲义 Table 1，默认上下文长度、词表和 batch 则来自 benchmark
题目的统一设置。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    """构造 ``BasicsTransformerLM`` 所需的全部结构参数。"""

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int

    def to_dict(self) -> dict[str, int]:
        """返回可直接写入 JSON 或传给模型构造函数的普通字典。"""

        return asdict(self)


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "tiny": ModelConfig(vocab_size=128, context_length=16, d_model=64, num_layers=2, num_heads=4, d_ff=128),
    "small": ModelConfig(vocab_size=10_000, context_length=512, d_model=768, num_layers=12, num_heads=12, d_ff=3_072),
    "medium": ModelConfig(vocab_size=10_000, context_length=512, d_model=1_024, num_layers=24, num_heads=16, d_ff=4_096),
    "large": ModelConfig(vocab_size=10_000, context_length=512, d_model=1_280, num_layers=36, num_heads=20, d_ff=5_120),
    "xl": ModelConfig(vocab_size=10_000, context_length=512, d_model=2_560, num_layers=32, num_heads=32, d_ff=10_240),
    "10b": ModelConfig(vocab_size=10_000, context_length=512, d_model=4_608, num_layers=50, num_heads=36, d_ff=12_288),
}


def resolve_model_config(
    model_size: str,
    *,
    vocab_size: int | None = None,
    context_length: int | None = None,
    d_model: int | None = None,
    num_layers: int | None = None,
    num_heads: int | None = None,
    d_ff: int | None = None,
) -> ModelConfig:
    """读取命名配置，并用显式命令行参数覆盖单个字段。

    覆盖入口用于 context-length 扫描和小尺寸排错。正式表格仍会同时记录
    ``model_size`` 与解析后的全部字段，所以覆盖不会隐藏真实实验配置。
    """

    try:
        base = MODEL_CONFIGS[model_size.lower()]
    except KeyError as exc:
        choices = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"unknown model size {model_size!r}; choose one of: {choices}") from exc

    overrides = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
    }
    return replace(base, **{name: value for name, value in overrides.items() if value is not None})
