# 本地自检脚本。它能抓大部分问题，最后还是得跑官方 pytest 才算数。

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from tests import adapters  # noqa: E402
from cs336_basics.model import Linear  # noqa: E402
from cs336_basics.training import train  # noqa: E402


def _snapshot(snapshot_dir: Path, name: str) -> np.ndarray:
    return np.load(snapshot_dir / f"{name}.npz")["array"]


def _assert_snapshot(actual: torch.Tensor, snapshot_dir: Path, name: str, atol=1e-5, rtol=1e-4) -> None:
    np.testing.assert_allclose(actual.detach().cpu().numpy(), _snapshot(snapshot_dir, name), atol=atol, rtol=rtol)


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    unicode_values = byte_values[:]
    offset = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, map(chr, unicode_values)))


def verify_model(official_root: Path) -> None:
    fixtures = official_root / "tests" / "fixtures"
    snapshots = official_root / "tests" / "_snapshots"
    state = torch.load(fixtures / "ts_tests" / "model.pt", map_location="cpu")
    state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
    batch_size, n_queries, n_keys = 4, 12, 16
    d_model, d_ff, num_heads, theta = 64, 128, 4, 10_000.0
    torch.manual_seed(4)
    embeddings = torch.randn(batch_size, n_queries, d_model)
    torch.manual_seed(6)
    indices = torch.randint(0, 10_000, (batch_size, n_queries))

    _assert_snapshot(
        adapters.run_linear(d_model, d_ff, state["layers.0.ffn.w1.weight"], embeddings),
        snapshots,
        "test_linear",
    )
    _assert_snapshot(
        adapters.run_embedding(10_000, d_model, state["token_embeddings.weight"], indices),
        snapshots,
        "test_embedding",
    )
    _assert_snapshot(
        adapters.run_swiglu(
            d_model,
            d_ff,
            state["layers.0.ffn.w1.weight"],
            state["layers.0.ffn.w2.weight"],
            state["layers.0.ffn.w3.weight"],
            embeddings,
        ),
        snapshots,
        "test_swiglu",
    )

    torch.manual_seed(1)
    queries = torch.randn(batch_size, n_queries, d_model)
    torch.manual_seed(2)
    keys = torch.randn(batch_size, n_keys, d_model)
    torch.manual_seed(3)
    values = torch.randn(batch_size, n_keys, d_model)
    torch.manual_seed(5)
    mask = torch.randn(batch_size, n_queries, n_keys) > 0.5
    _assert_snapshot(
        adapters.run_scaled_dot_product_attention(queries, keys, values, mask),
        snapshots,
        "test_scaled_dot_product_attention",
    )
    _assert_snapshot(
        adapters.run_scaled_dot_product_attention(
            queries.reshape(2, 2, n_queries, d_model),
            keys.reshape(2, 2, n_keys, d_model),
            values.reshape(2, 2, n_keys, d_model),
            mask.reshape(2, 2, n_queries, n_keys),
        ),
        snapshots,
        "test_4d_scaled_dot_product_attention",
    )

    projection_weights = [
        state[f"layers.0.attn.{name}_proj.weight"] for name in ("q", "k", "v", "output")
    ]
    _assert_snapshot(
        adapters.run_multihead_self_attention(
            d_model, num_heads, *projection_weights, in_features=embeddings
        ),
        snapshots,
        "test_multihead_self_attention",
    )
    positions = torch.arange(n_queries).reshape(1, n_queries)
    _assert_snapshot(
        adapters.run_multihead_self_attention_with_rope(
            d_model,
            num_heads,
            n_keys,
            theta,
            *projection_weights,
            in_features=embeddings,
            token_positions=positions,
        ),
        snapshots,
        "test_multihead_self_attention_with_rope",
    )
    _assert_snapshot(
        adapters.run_rope(d_model, theta, n_queries, embeddings, torch.arange(n_queries)),
        snapshots,
        "test_rope",
    )
    _assert_snapshot(
        adapters.run_rmsnorm(d_model, 1e-5, state["layers.1.ln1.weight"], embeddings),
        snapshots,
        "test_rmsnorm",
        atol=1e-4,
    )

    block_weights = {
        key.replace("layers.0.", ""): value for key, value in state.items() if key.startswith("layers.0.")
    }
    _assert_snapshot(
        adapters.run_transformer_block(
            d_model, num_heads, d_ff, n_keys, theta, block_weights, embeddings
        ),
        snapshots,
        "test_transformer_block",
        atol=1e-4,
    )
    _assert_snapshot(
        adapters.run_transformer_lm(10_000, n_keys, d_model, 3, num_heads, d_ff, theta, state, indices),
        snapshots,
        "test_transformer_lm",
        atol=1e-4,
        rtol=1e-2,
    )
    truncated = indices[..., : n_queries // 2]
    _assert_snapshot(
        adapters.run_transformer_lm(10_000, n_keys, d_model, 3, num_heads, d_ff, theta, state, truncated),
        snapshots,
        "test_transformer_lm_truncated_input",
        atol=1e-4,
    )


def verify_training_utilities() -> None:
    tensor = torch.tensor([[0.4, 0.8, 0.1], [100.0, 101.0, 99.0]])
    torch.testing.assert_close(adapters.run_softmax(tensor, -1), torch.softmax(tensor, -1))
    logits = torch.tensor([[1.0, 3.0, 2.0], [1000.0, 1002.0, 1001.0]])
    targets = torch.tensor([1, 2])
    expected_cross_entropy = (
        torch.logsumexp(logits, dim=-1) - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ).mean()
    torch.testing.assert_close(adapters.run_cross_entropy(logits, targets), expected_cross_entropy)

    parameters = [torch.nn.Parameter(torch.ones(3, 3)) for _ in range(2)]
    torch.stack([parameter.sum() for parameter in parameters]).sum().backward()
    adapters.run_gradient_clipping(parameters, 0.01)
    gradients = [parameter.grad for parameter in parameters]
    assert all(gradient is not None for gradient in gradients)
    total_norm = torch.stack(
        [gradient.square().sum() for gradient in gradients if gradient is not None]
    ).sum().sqrt()
    assert total_norm <= 0.010001

    data = np.arange(100)
    inputs, labels = adapters.run_get_batch(data, 16, 8, "cpu")
    assert inputs.shape == labels.shape == (16, 8)
    torch.testing.assert_close(inputs + 1, labels)

    expected_schedule = [0.0, 1 / 7, 1.0, 0.55, 0.1]
    schedule_steps = [0, 1, 7, 14, 21]
    actual_schedule = [adapters.run_get_lr_cosine_schedule(step, 1.0, 0.1, 7, 21) for step in schedule_steps]
    np.testing.assert_allclose(actual_schedule, expected_schedule)

    model = Linear(4, 2)
    optimizer = adapters.get_adamw_cls()(model.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(5):
        optimizer.zero_grad()
        model(torch.randn(3, 4)).square().mean().backward()
        optimizer.step()
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        adapters.run_save_checkpoint(model, optimizer, 5, checkpoint_path)
        restored_model = Linear(4, 2)
        restored_optimizer = adapters.get_adamw_cls()(restored_model.parameters(), lr=1e-3, weight_decay=0.01)
        assert adapters.run_load_checkpoint(checkpoint_path, restored_model, restored_optimizer) == 5
        for original, restored in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(original, restored)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        random = np.random.default_rng(0)
        np.save(temporary_root / "train.npy", random.integers(0, 32, 512, dtype=np.uint16))
        np.save(temporary_root / "valid.npy", random.integers(0, 32, 256, dtype=np.uint16))
        output_dir = temporary_root / "run"
        config = {
            "run_name": "resume_check",
            "dataset": "synthetic",
            "train_data": str(temporary_root / "train.npy"),
            "validation_data": str(temporary_root / "valid.npy"),
            "output_dir": str(output_dir),
            "vocab_size": 32,
            "context_length": 8,
            "d_model": 16,
            "d_ff": 32,
            "num_layers": 1,
            "num_heads": 4,
            "rope_theta": 10_000.0,
            "batch_size": 2,
            "total_steps": 2,
            "max_learning_rate": 0.001,
            "min_learning_rate": 0.0001,
            "warmup_steps": 1,
            "cosine_cycle_steps": 4,
            "max_grad_norm": 1.0,
            "log_interval": 1,
            "eval_interval": 1,
            "eval_batches": 1,
            "checkpoint_interval": 0,
            "device": "cpu",
            "seed": 0,
        }
        train(config)
        resumed_config = dict(config)
        resumed_config["total_steps"] = 4
        resumed_config["resume_from"] = str(output_dir / "checkpoint.pt")
        train(resumed_config)
        records = [json.loads(line) for line in (output_dir / "train.jsonl").read_text().splitlines()]
        assert [record["step"] for record in records] == [0, 1, 2, 3]
        wall_clocks = [record["wall_clock_sec"] for record in records]
        assert wall_clocks == sorted(wall_clocks)


def verify_tokenizer(official_root: Path) -> float:
    fixtures = official_root / "tests" / "fixtures"
    started = time.perf_counter()
    vocab, merges = adapters.run_train_bpe(fixtures / "corpus.en", 500, ["<|endoftext|>"])
    elapsed = time.perf_counter() - started

    byte_decoder = {value: key for key, value in _gpt2_bytes_to_unicode().items()}
    reference_merges = []
    for line in (fixtures / "train-bpe-reference-merges.txt").read_text(encoding="utf-8").splitlines():
        left, right = line.split(" ")
        reference_merges.append(
            (bytes(byte_decoder[character] for character in left), bytes(byte_decoder[character] for character in right))
        )
    assert merges == reference_merges
    reference_vocab_json = json.loads((fixtures / "train-bpe-reference-vocab.json").read_text(encoding="utf-8"))
    reference_vocab = {
        token_id: bytes(byte_decoder[character] for character in token)
        for token, token_id in reference_vocab_json.items()
    }
    assert set(vocab) == set(reference_vocab)
    assert set(vocab.values()) == set(reference_vocab.values())

    # 并行要是和串行不一样，那叫算错了，不叫优化。
    parallel_fixture = fixtures / "tinystories_sample.txt"
    serial_vocab, serial_merges = adapters.run_train_bpe(
        parallel_fixture,
        300,
        ["<|endoftext|>"],
        num_workers=1,
    )
    parallel_vocab, parallel_merges = adapters.run_train_bpe(
        parallel_fixture,
        300,
        ["<|endoftext|>"],
        num_workers=2,
        chunks_per_worker=2,
    )
    assert parallel_vocab == serial_vocab
    assert parallel_merges == serial_merges
    try:
        adapters.run_train_bpe(parallel_fixture, 300, ["<|endoftext|>"], num_workers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("num_workers=0 should be rejected")

    tokenizer = adapters.get_tokenizer(vocab, merges, ["<|endoftext|>"])
    sample = "A short story.<|endoftext|>第二个故事。"
    assert tokenizer.decode(tokenizer.encode(sample)) == sample
    arbitrary_chunks = ["A sh", "ort sto", "ry.<|endo", "ftext|>第二", "个故事。"]
    assert list(tokenizer.encode_iterable(arbitrary_chunks)) == tokenizer.encode(sample)
    overlap = adapters.get_tokenizer(
        vocab,
        merges,
        ["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    overlap_sample = "x<|endoftext|><|endoftext|>y<|endoftext|>"
    assert overlap.decode(overlap.encode(overlap_sample)) == overlap_sample
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    official_root = args.official_root.resolve()
    started = time.perf_counter()
    checks: dict[str, object] = {}
    verify_model(official_root)
    checks["model_and_attention_snapshots"] = "passed"
    verify_training_utilities()
    checks["training_utilities"] = "passed"
    bpe_seconds = verify_tokenizer(official_root)
    checks["tokenizer_and_bpe"] = "passed"
    checks["bpe_corpus_en_seconds"] = bpe_seconds
    payload = {
        "status": "passed",
        "wall_clock_sec": time.perf_counter() - started,
        "python": sys.version,
        "torch": torch.__version__,
        "checks": checks,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
