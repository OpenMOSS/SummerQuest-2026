from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast


NormMode = Literal["pre", "post", "none"]
FeedForwardType = Literal["swiglu", "silu"]
ModelDType = Literal["float32", "float16", "bfloat16"]
AmpDType = Literal["bfloat16"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    train_path: str
    validation_path: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.train_path, "data.train_path")
        _validate_non_empty_string(self.validation_path, "data.validation_path")


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    vocab_path: str
    merges_path: str
    special_tokens: tuple[str, ...]
    eot_token_id: int | None

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.vocab_path, "tokenizer.vocab_path")
        _validate_non_empty_string(self.merges_path, "tokenizer.merges_path")
        if any(not token for token in self.special_tokens):
            raise ValueError("tokenizer.special_tokens must not contain empty strings.")
        if self.eot_token_id is not None:
            _validate_non_negative_integer(self.eot_token_id, "tokenizer.eot_token_id")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    norm_mode: NormMode = "pre"
    use_rope: bool = True
    ffn_type: FeedForwardType = "swiglu"

    def __post_init__(self) -> None:
        _validate_positive_integer(self.vocab_size, "model.vocab_size")
        _validate_positive_integer(self.context_length, "model.context_length")
        _validate_positive_integer(self.d_model, "model.d_model")
        _validate_non_negative_integer(self.num_layers, "model.num_layers")
        _validate_positive_integer(self.num_heads, "model.num_heads")
        _validate_positive_integer(self.d_ff, "model.d_ff")
        if self.d_model % self.num_heads != 0:
            raise ValueError("model.d_model must be divisible by model.num_heads.")
        if self.use_rope and (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("RoPE requires an even per-head dimension.")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("model.rope_theta must be greater than zero.")
        if self.norm_mode not in ("pre", "post", "none"):
            raise ValueError("model.norm_mode must be one of: pre, post, none.")
        if self.ffn_type not in ("swiglu", "silu"):
            raise ValueError("model.ffn_type must be one of: swiglu, silu.")


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    cosine_cycle_steps: int
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_learning_rate) or not math.isfinite(self.min_learning_rate):
            raise ValueError("optimizer learning rates must be finite.")
        if self.max_learning_rate < 0 or self.min_learning_rate < 0:
            raise ValueError("optimizer learning rates must not be negative.")
        if self.min_learning_rate > self.max_learning_rate:
            raise ValueError("optimizer.min_learning_rate must not exceed max_learning_rate.")
        _validate_non_negative_integer(self.warmup_steps, "optimizer.warmup_steps")
        _validate_positive_integer(self.cosine_cycle_steps, "optimizer.cosine_cycle_steps")
        if self.cosine_cycle_steps <= self.warmup_steps:
            raise ValueError("optimizer.cosine_cycle_steps must be greater than warmup_steps.")
        beta1, beta2 = self.betas
        if not math.isfinite(beta1) or not math.isfinite(beta2) or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("optimizer.betas values must be in [0, 1).")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("optimizer.eps must be greater than zero.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must not be negative.")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("optimizer.max_grad_norm must be greater than zero.")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    max_wall_clock_sec: float | None
    eval_interval: int
    eval_batch_size: int
    eval_batches: int
    log_interval: int
    checkpoint_interval: int
    seed: int
    device: str
    dtype: ModelDType
    amp_dtype: AmpDType | None
    log_dir: str
    checkpoint_dir: str
    resume_from: str | None = None

    def __post_init__(self) -> None:
        _validate_positive_integer(self.batch_size, "training.batch_size")
        _validate_positive_integer(
            self.gradient_accumulation_steps,
            "training.gradient_accumulation_steps",
        )
        _validate_positive_integer(self.max_steps, "training.max_steps")
        if self.max_wall_clock_sec is not None and (
            not math.isfinite(self.max_wall_clock_sec) or self.max_wall_clock_sec <= 0
        ):
            raise ValueError("training.max_wall_clock_sec must be greater than zero when provided.")
        _validate_positive_integer(self.eval_interval, "training.eval_interval")
        _validate_positive_integer(self.eval_batch_size, "training.eval_batch_size")
        _validate_positive_integer(self.eval_batches, "training.eval_batches")
        _validate_positive_integer(self.log_interval, "training.log_interval")
        _validate_non_negative_integer(
            self.checkpoint_interval,
            "training.checkpoint_interval",
        )
        _validate_non_negative_integer(self.seed, "training.seed")
        _validate_non_empty_string(self.device, "training.device")
        _validate_non_empty_string(self.log_dir, "training.log_dir")
        _validate_non_empty_string(self.checkpoint_dir, "training.checkpoint_dir")
        if self.dtype != "float32":
            raise ValueError("training.dtype must be float32 so parameters and AdamW state remain full precision.")
        if self.amp_dtype not in (None, "bfloat16"):
            raise ValueError("training.amp_dtype must be null or bfloat16.")
        if self.resume_from is not None:
            _validate_non_empty_string(self.resume_from, "training.resume_from")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_name: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.experiment_name, "experiment_name")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load a JSON experiment config, resolving optional relative ``extends`` chains."""

    resolved_path = Path(config_path).expanduser().resolve()
    raw_config = _load_config_mapping(resolved_path, active_paths=set())
    return experiment_config_from_mapping(raw_config)


def experiment_config_from_mapping(raw_config: Mapping[str, Any]) -> ExperimentConfig:
    _require_exact_keys(
        raw_config,
        {"experiment_name", "data", "tokenizer", "model", "optimizer", "training"},
        "config",
    )

    data_mapping = _require_mapping(raw_config["data"], "data")
    tokenizer_mapping = _require_mapping(raw_config["tokenizer"], "tokenizer")
    model_mapping = _require_mapping(raw_config["model"], "model")
    optimizer_mapping = _require_mapping(raw_config["optimizer"], "optimizer")
    training_mapping = _require_mapping(raw_config["training"], "training")

    _require_exact_keys(data_mapping, {"train_path", "validation_path"}, "data")
    _require_exact_keys(
        tokenizer_mapping,
        {"vocab_path", "merges_path", "special_tokens", "eot_token_id"},
        "tokenizer",
    )
    _require_exact_keys(
        model_mapping,
        {
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
            "d_ff",
            "rope_theta",
            "norm_mode",
            "use_rope",
            "ffn_type",
        },
        "model",
    )
    _require_exact_keys(
        optimizer_mapping,
        {
            "max_learning_rate",
            "min_learning_rate",
            "warmup_steps",
            "cosine_cycle_steps",
            "betas",
            "eps",
            "weight_decay",
            "max_grad_norm",
        },
        "optimizer",
    )
    _require_exact_keys(
        training_mapping,
        {
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "max_wall_clock_sec",
            "eval_interval",
            "eval_batch_size",
            "eval_batches",
            "log_interval",
            "checkpoint_interval",
            "seed",
            "device",
            "dtype",
            "amp_dtype",
            "log_dir",
            "checkpoint_dir",
            "resume_from",
        },
        "training",
    )

    special_tokens_raw = tokenizer_mapping["special_tokens"]
    if not isinstance(special_tokens_raw, list) or not all(isinstance(token, str) for token in special_tokens_raw):
        raise TypeError("tokenizer.special_tokens must be a JSON array of strings.")

    betas_raw = optimizer_mapping["betas"]
    if not isinstance(betas_raw, list) or len(betas_raw) != 2:
        raise TypeError("optimizer.betas must be a JSON array containing two numbers.")

    return ExperimentConfig(
        experiment_name=_require_string(raw_config["experiment_name"], "experiment_name"),
        data=DataConfig(
            train_path=_require_string(data_mapping["train_path"], "data.train_path"),
            validation_path=_require_string(data_mapping["validation_path"], "data.validation_path"),
        ),
        tokenizer=TokenizerConfig(
            vocab_path=_require_string(tokenizer_mapping["vocab_path"], "tokenizer.vocab_path"),
            merges_path=_require_string(tokenizer_mapping["merges_path"], "tokenizer.merges_path"),
            special_tokens=tuple(special_tokens_raw),
            eot_token_id=_optional_integer(tokenizer_mapping["eot_token_id"], "tokenizer.eot_token_id"),
        ),
        model=ModelConfig(
            vocab_size=_require_integer(model_mapping["vocab_size"], "model.vocab_size"),
            context_length=_require_integer(model_mapping["context_length"], "model.context_length"),
            d_model=_require_integer(model_mapping["d_model"], "model.d_model"),
            num_layers=_require_integer(model_mapping["num_layers"], "model.num_layers"),
            num_heads=_require_integer(model_mapping["num_heads"], "model.num_heads"),
            d_ff=_require_integer(model_mapping["d_ff"], "model.d_ff"),
            rope_theta=_require_float(model_mapping["rope_theta"], "model.rope_theta"),
            norm_mode=cast(NormMode, _require_string(model_mapping["norm_mode"], "model.norm_mode")),
            use_rope=_require_bool(model_mapping["use_rope"], "model.use_rope"),
            ffn_type=cast(FeedForwardType, _require_string(model_mapping["ffn_type"], "model.ffn_type")),
        ),
        optimizer=OptimizerConfig(
            max_learning_rate=_require_float(
                optimizer_mapping["max_learning_rate"],
                "optimizer.max_learning_rate",
            ),
            min_learning_rate=_require_float(
                optimizer_mapping["min_learning_rate"],
                "optimizer.min_learning_rate",
            ),
            warmup_steps=_require_integer(optimizer_mapping["warmup_steps"], "optimizer.warmup_steps"),
            cosine_cycle_steps=_require_integer(
                optimizer_mapping["cosine_cycle_steps"],
                "optimizer.cosine_cycle_steps",
            ),
            betas=(
                _require_float(betas_raw[0], "optimizer.betas[0]"),
                _require_float(betas_raw[1], "optimizer.betas[1]"),
            ),
            eps=_require_float(optimizer_mapping["eps"], "optimizer.eps"),
            weight_decay=_require_float(optimizer_mapping["weight_decay"], "optimizer.weight_decay"),
            max_grad_norm=_require_float(optimizer_mapping["max_grad_norm"], "optimizer.max_grad_norm"),
        ),
        training=TrainingConfig(
            batch_size=_require_integer(training_mapping["batch_size"], "training.batch_size"),
            gradient_accumulation_steps=_require_integer(
                training_mapping["gradient_accumulation_steps"],
                "training.gradient_accumulation_steps",
            ),
            max_steps=_require_integer(training_mapping["max_steps"], "training.max_steps"),
            max_wall_clock_sec=_optional_float(
                training_mapping["max_wall_clock_sec"],
                "training.max_wall_clock_sec",
            ),
            eval_interval=_require_integer(training_mapping["eval_interval"], "training.eval_interval"),
            eval_batch_size=_require_integer(
                training_mapping["eval_batch_size"],
                "training.eval_batch_size",
            ),
            eval_batches=_require_integer(training_mapping["eval_batches"], "training.eval_batches"),
            log_interval=_require_integer(training_mapping["log_interval"], "training.log_interval"),
            checkpoint_interval=_require_integer(
                training_mapping["checkpoint_interval"],
                "training.checkpoint_interval",
            ),
            seed=_require_integer(training_mapping["seed"], "training.seed"),
            device=_require_string(training_mapping["device"], "training.device"),
            dtype=cast(ModelDType, _require_string(training_mapping["dtype"], "training.dtype")),
            amp_dtype=cast(
                AmpDType | None,
                _optional_string(training_mapping["amp_dtype"], "training.amp_dtype"),
            ),
            log_dir=_require_string(training_mapping["log_dir"], "training.log_dir"),
            checkpoint_dir=_require_string(training_mapping["checkpoint_dir"], "training.checkpoint_dir"),
            resume_from=_optional_string(training_mapping["resume_from"], "training.resume_from"),
        ),
    )


def _load_config_mapping(config_path: Path, active_paths: set[Path]) -> dict[str, Any]:
    if config_path in active_paths:
        chain = " -> ".join(str(path) for path in (*active_paths, config_path))
        raise ValueError(f"circular config inheritance detected: {chain}")
    if not config_path.is_file():
        raise FileNotFoundError(f"experiment config does not exist: {config_path}")

    try:
        raw_value = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in experiment config {config_path}: {error}") from error
    raw_mapping = dict(_require_mapping(raw_value, str(config_path)))
    extends = raw_mapping.pop("extends", None)
    if extends is None:
        return raw_mapping
    if not isinstance(extends, str) or not extends:
        raise TypeError(f"extends in {config_path} must be a non-empty string.")

    base_path = (config_path.parent / extends).resolve()
    active_paths.add(config_path)
    try:
        base_mapping = _load_config_mapping(base_path, active_paths)
    finally:
        active_paths.remove(config_path)
    return _deep_merge(base_mapping, raw_mapping)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _deep_merge(base_value, override_value)
        else:
            merged[key] = override_value
    return merged


def _require_exact_keys(mapping: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(mapping)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown: {', '.join(sorted(unknown))}")
        raise ValueError(f"invalid keys for {name} ({'; '.join(details)}).")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys.")
    return cast(Mapping[str, Any], value)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return value


def _optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _require_integer(value, name)


def _require_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    return float(value)


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    return _require_float(value, name)


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _validate_non_empty_string(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
