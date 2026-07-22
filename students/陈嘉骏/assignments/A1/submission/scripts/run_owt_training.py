from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer_experiments import load_tokenizer_artifact
from cs336_basics.training import TrainingConfig, load_token_dataset, train_language_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the required OpenWebText language model.")
    parser.add_argument("--config", default="configs/owt_baseline.json")
    parser.add_argument("--tokenizer", default="artifacts/owt_32k.json")
    parser.add_argument("--resume", help="Optional checkpoint to resume from.")
    parser.add_argument("--device", help="Override the configured device.")
    parser.add_argument("--batch-size", type=int, help="Override the configured batch size.")
    parser.add_argument("--max-learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--seed", type=int, help="Override the configured seed.")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _validate_encoded_dataset(path: Path, expected_vocab_size: int) -> dict[str, Any]:
    metadata_path = path.with_name(path.name + ".json")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing encoded dataset: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing encoded dataset metadata: {metadata_path}")

    metadata = _load_json(metadata_path)
    if metadata.get("vocab_size") != expected_vocab_size:
        raise ValueError(
            f"Dataset {path} uses vocab_size={metadata.get('vocab_size')}, expected {expected_vocab_size}."
        )
    token_count = metadata.get("token_count")
    if not isinstance(token_count, int) or token_count <= 0:
        raise ValueError(f"Invalid token_count in {metadata_path}.")
    expected_num_bytes = metadata.get("output_num_bytes")
    if expected_num_bytes != path.stat().st_size:
        raise ValueError(
            f"Dataset size mismatch for {path}: metadata={expected_num_bytes}, actual={path.stat().st_size}."
        )
    return metadata


def _resolve_training_config(configuration: dict[str, Any], args: argparse.Namespace) -> TrainingConfig:
    config = TrainingConfig(**configuration["training"])
    replacements: dict[str, object] = {}
    if args.device is not None:
        replacements["device"] = args.device
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        replacements["batch_size"] = args.batch_size
    if args.max_learning_rate is not None:
        replacements["max_learning_rate"] = args.max_learning_rate
    if args.min_learning_rate is not None:
        replacements["min_learning_rate"] = args.min_learning_rate
    if replacements:
        config = replace(config, **replacements)
    config.validate()
    return config


def main() -> None:
    args = _build_parser().parse_args()
    config_path = Path(args.config)
    configuration = _load_json(config_path)
    model_config: dict[str, Any] = configuration["model"]
    data_config: dict[str, Any] = configuration["data"]
    training_config = _resolve_training_config(configuration, args)
    if model_config["context_length"] != training_config.context_length:
        raise ValueError("model.context_length must match training.context_length.")

    tokenizer = load_tokenizer_artifact(args.tokenizer)
    if len(tokenizer.vocab) != model_config["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocab_size={len(tokenizer.vocab)} does not match model vocab_size={model_config['vocab_size']}."
        )

    train_path = Path(data_config["train"])
    validation_path = Path(data_config["validation"])
    train_metadata = _validate_encoded_dataset(train_path, model_config["vocab_size"])
    validation_metadata = _validate_encoded_dataset(validation_path, model_config["vocab_size"])
    if train_metadata["token_count"] <= training_config.context_length:
        raise ValueError("OWT training dataset is shorter than context_length.")
    if validation_metadata["token_count"] <= training_config.context_length:
        raise ValueError("OWT validation dataset is shorter than context_length.")

    seed = configuration.get("seed", 336) if args.seed is None else args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = TransformerLM(**model_config)
    train_dataset = load_token_dataset(train_path)
    validation_dataset = load_token_dataset(validation_path)
    summary = train_language_model(
        model,
        train_dataset,
        validation_dataset,
        training_config,
        resume_from=args.resume,
        run_metadata={
            "run_name": configuration.get("run_name", "owt_baseline"),
            "experiment": "openwebtext_training",
            "seed": seed,
            "tokenizer": os.fspath(args.tokenizer),
            "model": model_config,
            "data": data_config,
            "train_token_count": train_metadata["token_count"],
            "validation_token_count": validation_metadata["token_count"],
        },
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
