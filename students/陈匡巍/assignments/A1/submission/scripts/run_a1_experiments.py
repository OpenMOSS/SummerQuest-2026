from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.adapters_impl import (
    BPETokenizer,
    TransformerLM,
    get_adamw_cls,
    get_tokenizer,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_train_bpe,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_tokenizer(path: Path, vocab_size: int, special_tokens: list[str]) -> BPETokenizer:
    vocab, merges = run_train_bpe(path, vocab_size, special_tokens)
    return get_tokenizer(vocab, merges, special_tokens)


def load_or_train_tokenizer(
    path: Path,
    vocab_size: int,
    special_tokens: list[str],
    cache_path: Path,
    *,
    overwrite: bool,
) -> BPETokenizer:
    if cache_path.is_file() and not overwrite:
        with cache_path.open("rb") as f:
            return pickle.load(f)
    tokenizer = train_tokenizer(path, vocab_size, special_tokens)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(tokenizer, f)
    return tokenizer


def tokenizer_stats(tokenizer: BPETokenizer, text: str) -> dict:
    start = time.time()
    ids = tokenizer.encode(text)
    elapsed = max(time.time() - start, 1e-9)
    token_bytes = [tokenizer.vocab[index] for index in ids]
    return {
        "num_bytes": len(text.encode("utf-8")),
        "num_tokens": len(ids),
        "compression_ratio_bytes_per_token": len(text.encode("utf-8")) / max(len(ids), 1),
        "longest_token_bytes": max((len(token) for token in token_bytes), default=0),
        "throughput_bytes_per_sec": len(text.encode("utf-8")) / elapsed,
    }


def encode_text(tokenizer: BPETokenizer, text: str) -> np.ndarray:
    vocab_size = len(tokenizer.vocab)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    return np.array(tokenizer.encode(text), dtype=dtype)


def load_or_encode(cache_path: Path, tokenizer: BPETokenizer, text: str, *, overwrite: bool) -> np.ndarray:
    if cache_path.is_file() and not overwrite:
        return np.load(cache_path)
    ids = encode_text(tokenizer, text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, ids)
    return ids


def evaluate(model: TransformerLM, data: np.ndarray, batch_size: int, context_length: int, device: str, steps: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y = run_get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = run_cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            losses.append(float(loss.detach().cpu()))
    model.train()
    return float(sum(losses) / len(losses))


def train_run(
    *,
    name: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    log_path: Path,
    summary_path: Path,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    val_interval: int,
    val_batches: int,
    device: str,
    use_rmsnorm: bool = True,
    norm_position: str = "pre",
    use_rope: bool = True,
    ffn_variant: str = "swiglu",
    sample_tokenizer: BPETokenizer | None = None,
    sample_path: Path | None = None,
    sample_prompt: str = "Once upon a time",
    sample_note: str = "Sample generated from the trained checkpoint kept in memory during this script run.",
) -> dict:
    torch.manual_seed(20260713)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        use_rmsnorm=use_rmsnorm,
        norm_position=norm_position,
        use_rope=use_rope,
        ffn_variant=ffn_variant,
    ).to(device)
    optimizer = get_adamw_cls()(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    final_val_loss = None
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, total_steps + 1):
            lr = run_get_lr_cosine_schedule(step - 1, max_lr, min_lr, warmup_steps, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = run_get_batch(train_data, batch_size, context_length, device)
            logits = model(x)
            loss = run_cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            run_gradient_clipping(model.parameters(), 1.0)
            optimizer.step()
            record = {
                "step": step,
                "wall_clock_sec": round(time.time() - start, 4),
                "train_loss": float(loss.detach().cpu()),
                "lr": lr,
            }
            if step == 1 or step % val_interval == 0 or step == total_steps:
                final_val_loss = evaluate(model, val_data, batch_size, context_length, device, val_batches)
                record["val_loss"] = final_val_loss
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
    total_time = time.time() - start
    summary = {
        "name": name,
        "final_val_loss": final_val_loss,
        "total_train_time_sec": total_time,
        "processed_tokens": total_steps * batch_size * context_length,
        "config": {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "total_steps": total_steps,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "use_rmsnorm": use_rmsnorm,
            "norm_position": norm_position,
            "use_rope": use_rope,
            "ffn_variant": ffn_variant,
            "device": device,
        },
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        summary["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    if sample_tokenizer is not None and sample_path is not None:
        sample = {
            "prompt": sample_prompt,
            "sample": generate_sample(sample_tokenizer, model, sample_prompt, device),
            "note": sample_note,
        }
        write_json(sample_path, sample)
        summary["generation_sample_path"] = str(sample_path)
    write_json(summary_path, summary)
    return summary


def generate_sample(tokenizer: BPETokenizer, model: TransformerLM, prompt: str, device: str, max_new_tokens: int = 256) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    eos_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
            logits = model(context)[0, -1] / 0.8
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            keep = torch.cumsum(sorted_probs, dim=0) <= 0.95
            keep[0] = True
            filtered_probs = sorted_probs[keep]
            filtered_ids = sorted_ids[keep]
            next_id = int(filtered_ids[torch.multinomial(filtered_probs / filtered_probs.sum(), 1)].item())
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/a1_experiments"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tinystories", type=Path, default=Path("tests/fixtures/tinystories_sample_5M.txt"))
    parser.add_argument("--owt", type=Path, default=Path("tests/fixtures/corpus.en"))
    parser.add_argument("--max-chars", type=int, default=5_000_000)
    parser.add_argument("--tiny-max-chars", type=int)
    parser.add_argument("--owt-max-chars", type=int)
    parser.add_argument("--tiny-vocab-size", type=int, default=10_000)
    parser.add_argument("--owt-vocab-size", type=int, default=32_000)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=320)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=120)
    parser.add_argument("--max-lr", type=float, default=3e-3)
    parser.add_argument("--min-lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--val-interval", type=int, default=30)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--lr-values", default="0.001,0.003,0.01,0.03,0.1,1.0")
    parser.add_argument("--lr-sweep-steps", type=int, default=40)
    parser.add_argument("--batch-sizes", default="1,64,128")
    parser.add_argument("--batch-sweep-steps", type=int, default=30)
    parser.add_argument("--ablation-steps", type=int, default=50)
    parser.add_argument("--owt-steps", type=int)
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    tiny_max_chars = args.tiny_max_chars if args.tiny_max_chars is not None else args.max_chars
    owt_max_chars = args.owt_max_chars if args.owt_max_chars is not None else args.max_chars
    tiny_text = args.tinystories.read_text(encoding="utf-8")[:tiny_max_chars]
    owt_text = args.owt.read_text(encoding="utf-8")[:owt_max_chars]
    tiny_slice = output / "tinystories_slice.txt"
    owt_slice = output / "owt_slice.txt"
    tiny_slice.write_text(tiny_text, encoding="utf-8")
    owt_slice.write_text(owt_text, encoding="utf-8")

    tiny_tokenizer = load_or_train_tokenizer(
        tiny_slice,
        args.tiny_vocab_size,
        special_tokens,
        output / f"tinystories_tokenizer_v{args.tiny_vocab_size}_c{len(tiny_text)}.pkl",
        overwrite=args.overwrite_cache,
    )
    owt_tokenizer = load_or_train_tokenizer(
        owt_slice,
        args.owt_vocab_size,
        special_tokens,
        output / f"owt_tokenizer_v{args.owt_vocab_size}_c{len(owt_text)}.pkl",
        overwrite=args.overwrite_cache,
    )
    stats = {
        "tinystories": tokenizer_stats(tiny_tokenizer, tiny_text[:500_000]),
        "owt": tokenizer_stats(owt_tokenizer, owt_text[:500_000]),
        "slice_chars": {"tinystories": len(tiny_text), "owt": len(owt_text)},
    }
    write_json(output / "tokenizer_stats.json", stats)

    tiny_ids = load_or_encode(
        output / f"tinystories_ids_v{len(tiny_tokenizer.vocab)}_c{len(tiny_text)}.npy",
        tiny_tokenizer,
        tiny_text,
        overwrite=args.overwrite_cache,
    )
    split = int(len(tiny_ids) * 0.95)
    train_ids = tiny_ids[:split]
    val_ids = tiny_ids[split:]
    owt_ids = load_or_encode(
        output / f"owt_ids_v{len(owt_tokenizer.vocab)}_c{len(owt_text)}.npy",
        owt_tokenizer,
        owt_text,
        overwrite=args.overwrite_cache,
    )
    owt_split = max(int(len(owt_ids) * 0.9), 1)
    owt_train = owt_ids[:owt_split]
    owt_val = owt_ids[owt_split:]

    base = {
        "vocab_size": len(tiny_tokenizer.vocab),
        "context_length": args.context_length,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "warmup_steps": args.warmup_steps,
        "val_interval": args.val_interval,
        "val_batches": args.val_batches,
        "device": args.device,
    }
    def warmup_for(total_steps: int) -> int:
        return max(1, min(args.warmup_steps, total_steps // 10 if total_steps >= 10 else 1))

    summaries: dict[str, dict] = {}
    summaries["tinystories"] = train_run(
        name="tinystories",
        train_data=train_ids,
        val_data=val_ids,
        log_path=output / "train_tinystories.jsonl",
        summary_path=output / "summary_tinystories.json",
        sample_tokenizer=tiny_tokenizer,
        sample_path=output / "generation_sample.json",
        sample_prompt="Once upon a time",
        sample_note="Sample generated from the trained TinyStories checkpoint kept in memory during this script run.",
        **base,
    )
    lr_values = [float(value) for value in args.lr_values.split(",") if value.strip()]
    for lr in lr_values:
        cfg = dict(base)
        cfg.update(
            {
                "total_steps": args.lr_sweep_steps,
                "max_lr": lr,
                "min_lr": lr * 0.1,
                "warmup_steps": warmup_for(args.lr_sweep_steps),
                "val_interval": max(1, args.lr_sweep_steps // 4),
            }
        )
        summaries[f"lr_{lr:g}"] = train_run(
            name=f"lr_{lr:g}",
            train_data=train_ids,
            val_data=val_ids,
            log_path=output / "lr_sweep" / f"lr_{lr:g}.jsonl",
            summary_path=output / "lr_sweep" / f"summary_lr_{lr:g}.json",
            **cfg,
        )
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    for batch_size in batch_sizes:
        cfg = dict(base)
        cfg.update(
            {
                "total_steps": args.batch_sweep_steps,
                "batch_size": batch_size,
                "warmup_steps": warmup_for(args.batch_sweep_steps),
                "val_interval": max(1, args.batch_sweep_steps // 2),
            }
        )
        try:
            summaries[f"batch_{batch_size}"] = train_run(
                name=f"batch_{batch_size}",
                train_data=train_ids,
                val_data=val_ids,
                log_path=output / "batch_size" / f"batch_{batch_size}.jsonl",
                summary_path=output / "batch_size" / f"summary_batch_{batch_size}.json",
                **cfg,
            )
        except torch.OutOfMemoryError as error:
            if args.device.startswith("cuda") and torch.cuda.is_available():
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                torch.cuda.empty_cache()
            else:
                peak_memory_gb = None
            summary = {
                "name": f"batch_{batch_size}",
                "status": "oom",
                "error": str(error).splitlines()[0],
                "processed_tokens": 0,
                "peak_memory_gb": peak_memory_gb,
                "config": {
                    "vocab_size": cfg["vocab_size"],
                    "context_length": cfg["context_length"],
                    "d_model": cfg["d_model"],
                    "d_ff": cfg["d_ff"],
                    "num_layers": cfg["num_layers"],
                    "num_heads": cfg["num_heads"],
                    "batch_size": cfg["batch_size"],
                    "total_steps": cfg["total_steps"],
                    "max_lr": cfg["max_lr"],
                    "min_lr": cfg["min_lr"],
                    "warmup_steps": cfg["warmup_steps"],
                    "device": cfg["device"],
                },
            }
            write_json(output / "batch_size" / f"summary_batch_{batch_size}.json", summary)
            summaries[f"batch_{batch_size}"] = summary
    ablations = {
        "no_rmsnorm": {"use_rmsnorm": False},
        "post_norm": {"norm_position": "post"},
        "nope": {"use_rope": False},
        "silu_ffn": {"ffn_variant": "silu"},
    }
    for name, changes in ablations.items():
        cfg = dict(base)
        cfg.update(
            {
                "total_steps": args.ablation_steps,
                "warmup_steps": warmup_for(args.ablation_steps),
                "val_interval": max(1, args.ablation_steps // 5),
            }
        )
        if name == "silu_ffn":
            cfg["d_ff"] = int(round(base["d_ff"] * 1.5))
        cfg.update(changes)
        summaries[name] = train_run(
            name=name,
            train_data=train_ids,
            val_data=val_ids,
            log_path=output / f"ablation_{name}.jsonl",
            summary_path=output / f"summary_ablation_{name}.json",
            **cfg,
        )
    owt_cfg = dict(base)
    owt_cfg.update({"vocab_size": len(owt_tokenizer.vocab), "total_steps": args.owt_steps or args.total_steps})
    summaries["owt"] = train_run(
        name="owt",
        train_data=owt_train,
        val_data=owt_val,
        log_path=output / "train_owt.jsonl",
        summary_path=output / "summary_owt.json",
        sample_tokenizer=owt_tokenizer,
        sample_path=output / "generation_sample_owt.json",
        sample_prompt="The",
        sample_note="Sample generated from the trained OWT checkpoint kept in memory during this script run.",
        **owt_cfg,
    )
    sample = json.loads((output / "generation_sample.json").read_text(encoding="utf-8"))
    owt_sample = json.loads((output / "generation_sample_owt.json").read_text(encoding="utf-8"))
    write_json(output / "summary.json", {"tokenizer_stats": stats, "runs": summaries, "generation": sample, "generation_owt": owt_sample})


if __name__ == "__main__":
    main()
