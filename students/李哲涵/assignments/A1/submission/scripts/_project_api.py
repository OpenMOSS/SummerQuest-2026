from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

_PACKAGE = "cs336_basics"

# 常见文件名优先；若实现位于其他 cs336_basics/*.py，最后会自动扫描。
_PREFERRED_MODULES = (
    "cs336_basics.tokenizer",
    "cs336_basics.model",
    "cs336_basics.transformer",
    "cs336_basics.nn",
    "cs336_basics.optimizer",
    "cs336_basics.optim",
    "cs336_basics.training",
    "cs336_basics.train_utils",
    "cs336_basics.utils",
    "cs336_basics.serialization",
)

_IMPORTED_MODULES: dict[str, ModuleType] = {}
_SYMBOL_CACHE: dict[tuple[str, ...], Any] = {}
_DISCOVERED_MODULE_NAMES: list[str] | None = None


def _import_module(module_name: str) -> ModuleType | None:
    if module_name in _IMPORTED_MODULES:
        return _IMPORTED_MODULES[module_name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {module_name, _PACKAGE}:
            return None
        # 候选模块存在，但它内部缺少依赖；此时不能静默跳过。
        raise
    _IMPORTED_MODULES[module_name] = module
    return module


def _all_candidate_module_names() -> list[str]:
    global _DISCOVERED_MODULE_NAMES
    if _DISCOVERED_MODULE_NAMES is not None:
        return _DISCOVERED_MODULE_NAMES

    names = list(_PREFERRED_MODULES)
    package = _import_module(_PACKAGE)
    if package is None:
        raise RuntimeError(
            "找不到 cs336_basics。请从 assignment1-basics 仓库根目录运行脚本，"
            "例如：uv run python scripts/train_lm.py ..."
        )

    package_path = getattr(package, "__path__", None)
    if package_path is not None:
        for item in pkgutil.iter_modules(package_path, prefix=f"{_PACKAGE}."):
            if item.name not in names:
                names.append(item.name)

    _DISCOVERED_MODULE_NAMES = names
    return names


def resolve_symbol(*names: str) -> Any:
    key = tuple(names)
    if key in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[key]

    searched: list[str] = []
    for module_name in _all_candidate_module_names():
        module = _import_module(module_name)
        if module is None:
            continue
        searched.append(module_name)
        for name in names:
            if hasattr(module, name):
                value = getattr(module, name)
                _SYMBOL_CACHE[key] = value
                return value

    expected = " / ".join(names)
    raise RuntimeError(
        f"在 cs336_basics 中找不到 {expected}。已搜索：{', '.join(searched)}。"
        "请确认对应单测已经通过，或在 scripts/_project_api.py 中补充你的实际名称。"
    )


def get_train_bpe() -> Callable[..., Any]:
    return resolve_symbol("train_bpe")


def get_tokenizer_cls() -> type:
    return resolve_symbol("Tokenizer")


def get_model_cls() -> type:
    return resolve_symbol("TransformerLM", "TransformerLanguageModel")


def get_adamw_cls() -> type:
    return resolve_symbol("AdamW")


def get_batch_fn() -> Callable[..., Any]:
    return resolve_symbol("get_batch")


def get_cross_entropy_fn() -> Callable[..., Any]:
    return resolve_symbol("cross_entropy")


def get_gradient_clipping_fn() -> Callable[[Iterable[Any], float], None]:
    return resolve_symbol("gradient_clipping", "clip_gradients", "clip_grad_norm")


def get_lr_schedule_fn() -> Callable[..., float]:
    return resolve_symbol(
        "get_lr_cosine_schedule",
        "cosine_learning_rate_schedule",
        "get_cosine_lr",
    )


def get_save_checkpoint_fn() -> Callable[..., None]:
    return resolve_symbol("save_checkpoint")


def get_load_checkpoint_fn() -> Callable[..., int]:
    return resolve_symbol("load_checkpoint")


def build_model(config: dict[str, Any]) -> Any:
    """按构造函数签名映射参数，不在脚本中重写模型逻辑。"""
    model_cls = get_model_cls()
    signature = inspect.signature(model_cls)

    aliases: dict[str, tuple[str, ...]] = {
        "vocab_size": ("vocab_size",),
        "context_length": ("context_length", "max_seq_len", "max_seq_length"),
        "d_model": ("d_model",),
        "num_layers": ("num_layers", "n_layers"),
        "num_heads": ("num_heads", "n_heads"),
        "d_ff": ("d_ff", "ffn_dim", "intermediate_size"),
        "theta": ("rope_theta", "theta"),
        "norm_position": ("norm_position",),
        "ffn_type": ("ffn_type",),
    }

    kwargs: dict[str, Any] = {}
    for config_key, candidate_names in aliases.items():
        if config_key not in config:
            continue
        value = config[config_key]
        for candidate in candidate_names:
            if candidate in signature.parameters:
                kwargs[candidate] = value
                break

    optional_bools: dict[str, tuple[str, ...]] = {
        "use_rope": ("use_rope",),
        "use_final_norm": ("use_final_norm",),
    }
    derived_values = {
        "use_rope": not bool(config.get("no_rope", False)),
        "use_final_norm": not bool(config.get("no_final_norm", False)),
    }
    for config_key, candidate_names in optional_bools.items():
        for candidate in candidate_names:
            if candidate in signature.parameters:
                kwargs[candidate] = derived_values[config_key]
                break

    missing = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and name not in kwargs
    ]
    if missing:
        raise RuntimeError(
            f"无法为 {model_cls.__name__} 构造参数 {missing} 提供值。"
            "请在 scripts/_project_api.py 的 build_model() 中增加名称映射。"
        )

    return model_cls(**kwargs)


def build_optimizer(model: Any, config: dict[str, Any]) -> Any:
    optimizer_cls = get_adamw_cls()
    return optimizer_cls(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
        eps=float(config.get("eps", 1e-8)),
    )


def load_config_for_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    import json

    checkpoint = Path(checkpoint_path)
    config_path = checkpoint.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"找不到 {config_path}。本套脚本会在 checkpoint 同目录保存 config.json。"
        )
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)
