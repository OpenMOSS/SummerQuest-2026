"""CLI utilities: train_tokenizer, encode_corpus, sample."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .bpe import BPETokenizer, train_bpe
from .generate import generate
from .model import TransformerLM
from .optim import load_checkpoint, AdamW


def _save_tokenizer(vocab, merges, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tokenizer.pkl").open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)


def _load_tokenizer(dir_: str, special_tokens: list[str] | None = None) -> BPETokenizer:
    with open(Path(dir_) / "tokenizer.pkl", "rb") as f:
        d = pickle.load(f)
    return BPETokenizer(d["vocab"], d["merges"], special_tokens=special_tokens)


def cli_train_tokenizer():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_tokens, num_processes=args.processes)
    took = time.time() - t0
    out = Path(args.out)
    _save_tokenizer(vocab, merges, out)
    longest = max(vocab.values(), key=len)
    (out / "stats.json").write_text(json.dumps({
        "train_seconds": round(took, 2),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "longest_token_len": len(longest),
        "longest_token_hex": longest.hex(),
    }, indent=2))
    print(f"trained in {took:.1f}s -> {out}/tokenizer.pkl")


_MP_TOKENIZER = None  # per-process global for multiprocessing


def _mp_init(vocab, merges, specials):
    global _MP_TOKENIZER
    _MP_TOKENIZER = BPETokenizer(vocab, merges, specials)


def _mp_encode_chunk(args):
    """Encode a byte range of the file and return raw ids as bytes for the given dtype."""
    path, start, end, dtype_str = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    ids = _MP_TOKENIZER.encode(text)
    import numpy as _np
    dtype = _np.uint16 if dtype_str == "uint16" else _np.uint32
    return _np.asarray(ids, dtype=dtype).tobytes()


def cli_encode_corpus():
    from concurrent.futures import ProcessPoolExecutor
    from .bpe import find_chunk_boundaries

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer.pkl")
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    ap.add_argument("--processes", type=int, default=1)
    args = ap.parse_args()
    tok = _load_tokenizer(args.tokenizer, args.special_tokens)
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    max_id = max(tok.vocab.keys())
    assert max_id < np.iinfo(dtype).max, f"vocab too large for {args.dtype}"
    t0 = time.time()
    total_bytes = Path(args.input).stat().st_size
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.processes > 1:
        # Chunk on <|endoftext|> boundaries so each worker sees complete documents.
        split_tok = args.special_tokens[0].encode("utf-8") if args.special_tokens else b"\n"
        with open(args.input, "rb") as f:
            bounds = find_chunk_boundaries(f, args.processes * 8, split_tok)
        tasks = [(args.input, s, e, args.dtype) for s, e in zip(bounds[:-1], bounds[1:])]
        n_tokens = 0
        with ProcessPoolExecutor(
            max_workers=args.processes,
            initializer=_mp_init,
            initargs=(tok.vocab, tok.merges, args.special_tokens),
        ) as ex, open(out_path, "wb") as fout:
            for buf in ex.map(_mp_encode_chunk, tasks):
                fout.write(buf)
                n_tokens += len(buf) // dtype().itemsize
    else:
        n_tokens = 0
        with open(args.input) as fin, open(out_path, "wb") as fout:
            arr_chunk = []
            for tid in tok.encode_iterable(fin):
                arr_chunk.append(tid)
                if len(arr_chunk) >= 1_000_000:
                    np.asarray(arr_chunk, dtype=dtype).tofile(fout)
                    n_tokens += len(arr_chunk)
                    arr_chunk = []
            if arr_chunk:
                np.asarray(arr_chunk, dtype=dtype).tofile(fout)
                n_tokens += len(arr_chunk)
    took = time.time() - t0
    stats = {
        "input_bytes": total_bytes,
        "num_tokens": n_tokens,
        "compression_bytes_per_token": total_bytes / max(1, n_tokens),
        "seconds": round(took, 2),
        "throughput_bytes_per_sec": round(total_bytes / max(1e-6, took), 0),
    }
    (out_path.with_suffix(out_path.suffix + ".stats.json")).write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def cli_sample():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--config", required=True, help="path to config.json from training")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    ap.add_argument("--eos", default="<|endoftext|>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=1, help="number of samples")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    tok = _load_tokenizer(args.tokenizer, args.special_tokens)
    model = TransformerLM(
        cfg["vocab_size"], cfg["context_length"], cfg["d_model"], cfg["num_layers"],
        cfg["num_heads"], cfg["d_ff"], cfg["rope_theta"], device=args.device,
    )
    dummy = AdamW(model.parameters(), lr=0.0)  # load_checkpoint expects an optimizer
    load_checkpoint(args.ckpt, model, dummy)
    model.to(args.device)

    eos_id = tok._bytes_to_id.get(args.eos.encode("utf-8")) if args.eos else None
    prompt_ids = tok.encode(args.prompt) if args.prompt else []
    if not prompt_ids:
        prompt_ids = [eos_id] if eos_id is not None else [0]
    for i in range(args.n):
        ids = generate(model, prompt_ids, args.max_new_tokens,
                       temperature=args.temperature, top_p=args.top_p,
                       eos_id=eos_id, device=args.device)
        text = tok.decode(ids)
        print(f"===== sample {i} =====\n{text}\n")


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    {"train_tokenizer": cli_train_tokenizer,
     "encode_corpus": cli_encode_corpus,
     "sample": cli_sample}[cmd]()
