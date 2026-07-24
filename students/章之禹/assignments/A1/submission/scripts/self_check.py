"""Run deterministic CPU checks for the A1 optimizer/training/generation stack."""

from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.generation import generate, sample_next_token
from cs336_basics.optim import AdamW, clip_gradients, cross_entropy, get_lr_cosine_schedule
from cs336_basics.training import build_model, get_batch, load_checkpoint, load_config, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional TOML config; uses [model] and [self_check]")
    parser.add_argument("--output", help="Optional JSON report path")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "model": {
            "vocab_size": 128,
            "context_length": 32,
            "d_model": 64,
            "num_layers": 2,
            "num_heads": 4,
            "d_ff": 192,
            "rope_theta": 10_000.0,
        },
        "training": {"device": "cpu"},
        "self_check": {"batch_size": 4, "steps": 100, "learning_rate": 3e-3, "seed": 1337},
    }


def _optimizer_step(model: torch.nn.Module, optimizer: AdamW, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = cross_entropy(model(inputs), targets)
    loss.backward()
    clip_gradients(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach())


def _assert_models_close(left: torch.nn.Module, right: torch.nn.Module) -> None:
    left_state, right_state = left.state_dict(), right.state_dict()
    if left_state.keys() != right_state.keys():
        raise AssertionError("checkpoint round-trip changed model state keys")
    for key in left_state:
        torch.testing.assert_close(left_state[key], right_state[key], rtol=1e-6, atol=1e-7, msg=lambda msg: f"{key}: {msg}")


def _core_math_checks() -> dict[str, Any]:
    logits = torch.tensor([[1000.0, 1001.0, 999.0], [-1000.0, -999.0, -1001.0]], requires_grad=True)
    targets = torch.tensor([1, 0])
    loss = cross_entropy(logits, targets)
    expected = (torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets[:, None]).squeeze(-1)).mean()
    torch.testing.assert_close(loss, expected, rtol=2e-5, atol=2e-5)
    loss.backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise AssertionError("stable cross-entropy produced non-finite gradients")

    expected_schedule = (0.0, 1.0, 0.1, 0.1)
    actual_schedule = (
        get_lr_cosine_schedule(0, 1.0, 0.1, 2, 6),
        get_lr_cosine_schedule(2, 1.0, 0.1, 2, 6),
        get_lr_cosine_schedule(6, 1.0, 0.1, 2, 6),
        get_lr_cosine_schedule(7, 1.0, 0.1, 2, 6),
    )
    np.testing.assert_allclose(actual_schedule, expected_schedule)

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    parameter.grad = torch.tensor([0.5])
    optimizer = AdamW([parameter], lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
    optimizer.step()
    # First-step hand calculation from the v26.0.3 pseudocode.
    decayed = 2.0 - 0.01 * 0.1 * 2.0
    m, v = 0.05, 0.00025
    adjusted_lr = 0.01 * (1 - 0.999) ** 0.5 / (1 - 0.9)
    expected_parameter = decayed - adjusted_lr * m / (v**0.5 + 1e-8)
    torch.testing.assert_close(parameter, torch.tensor([expected_parameter]), rtol=1e-6, atol=1e-7)
    return {"cross_entropy": float(loss.detach()), "adamw_first_step": float(parameter.detach())}


def _variant_checks(base_config: dict[str, Any], seed: int) -> list[str]:
    variants: dict[str, dict[str, Any]] = {
        "no_rmsnorm": {"remove_rmsnorm": True},
        "post_norm": {"use_post_norm": True},
        "no_rope": {"remove_rope": True},
        "silu_ffn": {"ffn_type": "silu", "silu_d_ff": int(base_config["model"]["d_ff"] * 1.5)},
    }
    passed: list[str] = []
    for offset, (name, overrides) in enumerate(variants.items()):
        torch.manual_seed(seed + 100 + offset)
        config = copy.deepcopy(base_config)
        config["model"].update(overrides)
        model = build_model(config, "cpu")
        inputs = torch.randint(
            0,
            int(config["model"]["vocab_size"]),
            (2, min(8, int(config["model"]["context_length"]))),
        )
        loss = model(inputs).square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise AssertionError(f"{name} produced a non-finite forward/backward result")
        passed.append(name)
    return passed


def run_self_check(config: dict[str, Any]) -> dict[str, Any]:
    self_check_cfg = dict(config.get("self_check", {}))
    seed = int(self_check_cfg.get("seed", 1337))
    batch_size = int(self_check_cfg.get("batch_size", 4))
    steps = int(self_check_cfg.get("steps", 100))
    learning_rate = float(self_check_cfg.get("learning_rate", 3e-3))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(self_check_cfg.get("num_threads", 1)))

    math_report = _core_math_checks()
    model = build_model(config, "cpu")
    model_cfg = config["model"]
    dataset = np.arange(int(model_cfg["vocab_size"]), dtype=np.int64)
    dataset = np.resize(dataset, max(4096, int(model_cfg["context_length"]) + 1))
    batch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    inputs, targets = get_batch(
        dataset,
        batch_size=batch_size,
        context_length=int(model_cfg["context_length"]),
        device="cpu",
        generator=batch_generator,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    with torch.no_grad():
        initial_loss = float(cross_entropy(model(inputs), targets))
    for _ in range(steps):
        _optimizer_step(model, optimizer, inputs, targets)
    with torch.no_grad():
        final_loss = float(cross_entropy(model(inputs), targets))
    if not final_loss <= 0.5 * initial_loss:
        raise AssertionError(
            f"fixed-batch loss did not fall by at least 50%: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )

    # Checkpoint a live optimizer, then verify one continued step is identical
    # before and after restoration.
    checkpoint = io.BytesIO()
    save_checkpoint(model, optimizer, iteration=steps, out=checkpoint)
    expected_model = copy.deepcopy(model)
    expected_optimizer = AdamW(
        expected_model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    expected_optimizer.load_state_dict(optimizer.state_dict())
    _optimizer_step(expected_model, expected_optimizer, inputs, targets)

    restored_model = build_model(config, "cpu")
    restored_optimizer = AdamW(
        restored_model.parameters(), lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    checkpoint.seek(0)
    restored_iteration = load_checkpoint(checkpoint, restored_model, restored_optimizer)
    if restored_iteration != steps:
        raise AssertionError("checkpoint restored the wrong iteration")
    _optimizer_step(restored_model, restored_optimizer, inputs, targets)
    _assert_models_close(expected_model, restored_model)

    greedy = generate(model, inputs[0, :4], max_new_tokens=4, temperature=0, top_p=1.0)
    if greedy.shape != (8,):
        raise AssertionError(f"generation returned unexpected shape {tuple(greedy.shape)}")
    sampled = sample_next_token(
        torch.tensor([0.0, 1.0, 2.0]),
        temperature=1.0,
        top_p=0.5,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if int(sampled) != 2:
        raise AssertionError("top-p failed to retain the minimum high-probability prefix")

    variants = _variant_checks(config, seed)
    return {
        "status": "passed",
        "seed": seed,
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "checkpoint_continuation": "passed",
        "generation": "passed",
        "variants": variants,
        **math_report,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else _default_config()
    report = run_self_check(config)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
