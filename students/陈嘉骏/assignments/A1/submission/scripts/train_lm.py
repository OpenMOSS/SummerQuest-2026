from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import TrainingConfig, load_token_dataset, train_language_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the from-scratch Transformer language model.")
    parser.add_argument("--config", required=True, help="JSON file containing model, data, and training sections.")
    parser.add_argument("--resume", help="Optional checkpoint to resume from.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with open(args.config, encoding="utf-8") as f:
        configuration = json.load(f)

    model_config = configuration["model"]
    data_config = configuration["data"]
    training_config = TrainingConfig(**configuration["training"])
    seed = configuration.get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if model_config["context_length"] != training_config.context_length:
        raise ValueError("model.context_length must match training.context_length.")

    model = TransformerLM(**model_config)
    train_dataset = load_token_dataset(data_config["train"])
    validation_path = data_config.get("validation")
    validation_dataset = load_token_dataset(validation_path) if validation_path is not None else None
    summary = train_language_model(
        model,
        train_dataset,
        validation_dataset,
        training_config,
        resume_from=args.resume,
        run_metadata={
            "run_name": configuration.get("run_name", "language_model_training"),
            "seed": seed,
            "model": model_config,
            "data": data_config,
        },
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
