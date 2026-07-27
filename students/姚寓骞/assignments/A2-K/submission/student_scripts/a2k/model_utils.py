"""Compatibility helpers for the assignment1 and bundled basics models."""

from __future__ import annotations

from cs336_basics import model as basics_model


def build_transformer(vocab_size: int, context_length: int, config: dict):
    """Build the Stanford model across the two class-name variants in this workspace."""
    if hasattr(basics_model, "BasicsTransformerLM"):
        model_class = basics_model.BasicsTransformerLM
        return model_class(vocab_size=vocab_size, context_length=context_length, **config)
    if hasattr(basics_model, "TransformerLM"):
        model_class = basics_model.TransformerLM
        return model_class(
            vocab_size=vocab_size,
            context_length=context_length,
            rope_theta=10_000.0,
            **config,
        )
    raise ImportError("cs336_basics.model exposes neither BasicsTransformerLM nor TransformerLM")


__all__ = ["build_transformer"]
