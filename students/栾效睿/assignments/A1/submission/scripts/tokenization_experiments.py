from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cs336_basics.bpe_tokenizer import BPETokenizer


DEFAULT_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_READ_SIZE = 16 * 1024 * 1024
DEFAULT_BATCH_BYTES = 8 * 1024 * 1024
DEFAULT_SUITE_CONFIG = ROOT / "configs" / "tokenizer_experiments.json"

_WORKER_TOKENIZER: BPETokenizer | None = None


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    vocab_path: Path
    merges_path: Path


@dataclass(frozen=True)
class Document:
    data: bytes
    ended_with_delimiter: bool


@dataclass(frozen=True)
class BatchStats:
    index: int
    byte_count: int
    token_count: int


def iter_documents(
    input_path: Path,
    special_token: str,
    read_size: int = DEFAULT_READ_SIZE,
    max_docs: int | None = None,
    max_bytes: int | None = None,
) -> Iterator[Document]:
    delimiter = special_token.encode("utf-8")
    remainder = b""
    emitted_docs = 0
    bytes_read = 0

    with input_path.open("rb") as input_file:
        while True:
            if max_bytes is not None:
                remaining_bytes = max_bytes - bytes_read
                if remaining_bytes <= 0:
                    break
                chunk = input_file.read(min(read_size, remaining_bytes))
            else:
                chunk = input_file.read(read_size)

            if not chunk:
                break

            bytes_read += len(chunk)
            data = remainder + chunk
            parts = data.split(delimiter)
            remainder = parts.pop()

            for part in parts:
                if max_docs is not None and emitted_docs >= max_docs:
                    return
                emitted_docs += 1
                yield Document(part, ended_with_delimiter=True)

    if remainder and (max_docs is None or emitted_docs < max_docs):
        yield Document(remainder, ended_with_delimiter=False)


def sample_documents(
    input_path: Path,
    special_token: str,
    sample_docs: int,
    seed: int,
    max_docs: int | None,
    max_bytes: int | None,
) -> list[Document]:
    rng = random.Random(seed)
    reservoir: list[Document] = []
    seen = 0

    for document in iter_documents(
        input_path=input_path,
        special_token=special_token,
        max_docs=max_docs,
        max_bytes=max_bytes,
    ):
        if not document.data:
            continue
        seen += 1
        if len(reservoir) < sample_docs:
            reservoir.append(document)
            continue

        replacement_index = rng.randrange(seen)
        if replacement_index < sample_docs:
            reservoir[replacement_index] = document

    return reservoir


def parse_tokenizer_spec(raw_spec: str) -> TokenizerSpec:
    parts = raw_spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Tokenizer spec must be NAME:VOCAB_PATH:MERGES_PATH"
        )
    name, vocab_path, merges_path = parts
    return TokenizerSpec(name=name, vocab_path=Path(vocab_path), merges_path=Path(merges_path))


def load_tokenizer(spec: TokenizerSpec, special_token: str) -> BPETokenizer:
    return BPETokenizer.from_files(
        spec.vocab_path,
        spec.merges_path,
        special_tokens=[special_token],
    )


def init_tokenizer_worker(spec: TokenizerSpec, special_token: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = load_tokenizer(spec, special_token)


def get_worker_tokenizer() -> BPETokenizer:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Tokenizer worker was not initialized")
    return _WORKER_TOKENIZER


def document_text(document: Document, special_token: str, include_delimiter: bool) -> str:
    text = document.data.decode("utf-8", errors="replace")
    if include_delimiter and document.ended_with_delimiter:
        return text + special_token
    return text


def document_bytes(document: Document, special_token: str, include_delimiter: bool) -> int:
    byte_count = len(document.data)
    if include_delimiter and document.ended_with_delimiter:
        byte_count += len(special_token.encode("utf-8"))
    return byte_count


def iter_text_batches(
    input_path: Path,
    special_token: str,
    include_delimiter: bool,
    target_batch_bytes: int,
    max_docs: int | None,
    max_bytes: int | None,
) -> Iterator[tuple[str, int]]:
    parts: list[str] = []
    batch_bytes = 0

    for document in iter_documents(
        input_path=input_path,
        special_token=special_token,
        max_docs=max_docs,
        max_bytes=max_bytes,
    ):
        text = document_text(document, special_token, include_delimiter)
        byte_count = document_bytes(document, special_token, include_delimiter)
        if parts and batch_bytes + byte_count > target_batch_bytes:
            yield "".join(parts), batch_bytes
            parts = []
            batch_bytes = 0
        parts.append(text)
        batch_bytes += byte_count

    if parts:
        yield "".join(parts), batch_bytes


def iter_indexed_text_batches(
    input_path: Path,
    special_token: str,
    include_delimiter: bool,
    target_batch_bytes: int,
    max_docs: int | None,
    max_bytes: int | None,
) -> Iterator[tuple[int, str, int]]:
    for batch_index, (text, byte_count) in enumerate(
        iter_text_batches(
            input_path=input_path,
            special_token=special_token,
            include_delimiter=include_delimiter,
            target_batch_bytes=target_batch_bytes,
            max_docs=max_docs,
            max_bytes=max_bytes,
        ),
        start=1,
    ):
        yield batch_index, text, byte_count


def compute_ratio(
    tokenizer: BPETokenizer,
    documents: list[Document],
    special_token: str,
    include_delimiter: bool,
) -> tuple[int, int, float, list[dict[str, object]]]:
    total_bytes = 0
    total_tokens = 0
    per_document: list[dict[str, object]] = []
    for document_index, document in enumerate(documents):
        text = document_text(document, special_token, include_delimiter)
        token_ids = tokenizer.encode(text)
        byte_count = document_bytes(document, special_token, include_delimiter)
        token_count = len(token_ids)
        total_bytes += byte_count
        total_tokens += token_count
        per_document.append(
            {
                "document_index": document_index,
                "bytes": byte_count,
                "tokens": token_count,
                "token_ids": token_ids,
            }
        )

    ratio = total_bytes / total_tokens if total_tokens else float("nan")
    return total_bytes, total_tokens, ratio, per_document


def run_ratio(args: argparse.Namespace) -> None:
    documents = sample_documents(
        input_path=args.input_path,
        special_token=args.special_token,
        sample_docs=args.sample_docs,
        seed=args.seed,
        max_docs=args.max_docs,
        max_bytes=args.max_bytes,
    )
    if not documents:
        raise ValueError("No documents were sampled")

    results = {
        "input_path": str(args.input_path),
        "sample_docs": len(documents),
        "include_delimiter": args.include_delimiter,
        "seed": args.seed,
        "tokenizers": [],
    }

    for spec in args.tokenizer:
        tokenizer = load_tokenizer(spec, args.special_token)
        total_bytes, total_tokens, ratio, per_document = compute_ratio(
            tokenizer=tokenizer,
            documents=documents,
            special_token=args.special_token,
            include_delimiter=args.include_delimiter,
        )
        item = {
            "name": spec.name,
            "vocab_path": str(spec.vocab_path),
            "merges_path": str(spec.merges_path),
            "bytes": total_bytes,
            "tokens": total_tokens,
            "bytes_per_token": ratio,
        }
        if args.include_token_ids:
            item["documents"] = per_document
        results["tokenizers"].append(item)
        print(
            f"{spec.name}: bytes={total_bytes} tokens={total_tokens} "
            f"bytes/token={ratio:.4f}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def validate_uint16_vocab(tokenizer: BPETokenizer) -> None:
    max_token_id = max(tokenizer.vocab.keys(), default=-1)
    if max_token_id > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Tokenizer max token id {max_token_id} exceeds uint16 range"
        )


def count_encoded_tokens(
    tokenizer: BPETokenizer,
    input_path: Path,
    special_token: str,
    include_delimiter: bool,
    max_docs: int | None,
    max_bytes: int | None,
    batch_bytes: int,
    progress_every_batches: int,
) -> tuple[int, int]:
    total_tokens = 0
    total_bytes = 0
    for batch_index, (text, byte_count) in enumerate(
        iter_text_batches(
            input_path=input_path,
            special_token=special_token,
            include_delimiter=include_delimiter,
            target_batch_bytes=batch_bytes,
            max_docs=max_docs,
            max_bytes=max_bytes,
        ),
        start=1,
    ):
        total_tokens += len(tokenizer.encode(text))
        total_bytes += byte_count
        if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
            print(
                f"counted batch {batch_index}: bytes={total_bytes} tokens={total_tokens}",
                flush=True,
            )
    return total_tokens, total_bytes


def count_batch_worker(task: tuple[int, str, int]) -> BatchStats:
    batch_index, text, byte_count = task
    token_count = len(get_worker_tokenizer().encode(text))
    return BatchStats(index=batch_index, byte_count=byte_count, token_count=token_count)


def count_encoded_tokens_parallel(
    tokenizer_spec: TokenizerSpec,
    input_path: Path,
    special_token: str,
    include_delimiter: bool,
    max_docs: int | None,
    max_bytes: int | None,
    batch_bytes: int,
    workers: int,
    max_inflight_batches: int,
    progress_every_batches: int,
) -> list[BatchStats]:
    stats: list[BatchStats] = []
    pending = set()
    tasks = iter(
        iter_indexed_text_batches(
            input_path=input_path,
            special_token=special_token,
            include_delimiter=include_delimiter,
            target_batch_bytes=batch_bytes,
            max_docs=max_docs,
            max_bytes=max_bytes,
        )
    )
    completed_batches = 0
    total_tokens = 0
    total_bytes = 0
    tasks_exhausted = False

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_tokenizer_worker,
        initargs=(tokenizer_spec, special_token),
    ) as executor:
        while pending or not tasks_exhausted:
            while not tasks_exhausted and len(pending) < max_inflight_batches:
                try:
                    pending.add(executor.submit(count_batch_worker, next(tasks)))
                except StopIteration:
                    tasks_exhausted = True

            if not pending:
                continue

            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                stat = future.result()
                stats.append(stat)
                completed_batches += 1
                total_tokens += stat.token_count
                total_bytes += stat.byte_count
                if progress_every_batches > 0 and completed_batches % progress_every_batches == 0:
                    print(
                        f"counted {completed_batches} batches: "
                        f"bytes={total_bytes} tokens={total_tokens}",
                        flush=True,
                    )

    return sorted(stats, key=lambda stat: stat.index)


def write_encoded_tokens(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    special_token: str,
    include_delimiter: bool,
    max_docs: int | None,
    max_bytes: int | None,
    total_tokens: int,
    batch_bytes: int,
    progress_every_batches: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    offset = 0
    for batch_index, (text, _) in enumerate(
        iter_text_batches(
            input_path=input_path,
            special_token=special_token,
            include_delimiter=include_delimiter,
            target_batch_bytes=batch_bytes,
            max_docs=max_docs,
            max_bytes=max_bytes,
        ),
        start=1,
    ):
        token_ids = tokenizer.encode(text)
        next_offset = offset + len(token_ids)
        token_array[offset:next_offset] = np.asarray(token_ids, dtype=np.uint16)
        offset = next_offset
        if progress_every_batches > 0 and batch_index % progress_every_batches == 0:
            print(f"wrote batch {batch_index}: tokens={offset}", flush=True)

    token_array.flush()


def write_batch_worker(task: tuple[int, str, int, int, str]) -> BatchStats:
    batch_index, text, byte_count, offset, output_path = task
    token_ids = get_worker_tokenizer().encode(text)
    token_array = np.lib.format.open_memmap(output_path, mode="r+")
    next_offset = offset + len(token_ids)
    token_array[offset:next_offset] = np.asarray(token_ids, dtype=np.uint16)
    token_array.flush()
    return BatchStats(index=batch_index, byte_count=byte_count, token_count=len(token_ids))


def write_encoded_tokens_parallel(
    tokenizer_spec: TokenizerSpec,
    input_path: Path,
    output_path: Path,
    special_token: str,
    include_delimiter: bool,
    max_docs: int | None,
    max_bytes: int | None,
    batch_bytes: int,
    workers: int,
    max_inflight_batches: int,
    progress_every_batches: int,
    batch_stats: list[BatchStats],
    total_tokens: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_array = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )
    token_array.flush()

    offsets: dict[int, int] = {}
    expected_stats: dict[int, BatchStats] = {}
    offset = 0
    for stat in batch_stats:
        offsets[stat.index] = offset
        expected_stats[stat.index] = stat
        offset += stat.token_count
    if offset != total_tokens:
        raise ValueError(f"Batch token counts sum to {offset}, expected {total_tokens}")

    pending = set()
    tasks = iter(
        (
            batch_index,
            text,
            byte_count,
            offsets[batch_index],
            str(output_path),
        )
        for batch_index, text, byte_count in iter_indexed_text_batches(
            input_path=input_path,
            special_token=special_token,
            include_delimiter=include_delimiter,
            target_batch_bytes=batch_bytes,
            max_docs=max_docs,
            max_bytes=max_bytes,
        )
    )
    completed_batches = 0
    completed_tokens = 0
    tasks_exhausted = False

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_tokenizer_worker,
        initargs=(tokenizer_spec, special_token),
    ) as executor:
        while pending or not tasks_exhausted:
            while not tasks_exhausted and len(pending) < max_inflight_batches:
                try:
                    pending.add(executor.submit(write_batch_worker, next(tasks)))
                except StopIteration:
                    tasks_exhausted = True

            if not pending:
                continue

            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                stat = future.result()
                expected = expected_stats[stat.index]
                if stat.token_count != expected.token_count:
                    raise ValueError(
                        f"Batch {stat.index} changed token count from "
                        f"{expected.token_count} to {stat.token_count}"
                    )
                completed_batches += 1
                completed_tokens += stat.token_count
                if progress_every_batches > 0 and completed_batches % progress_every_batches == 0:
                    print(
                        f"wrote {completed_batches} batches: tokens={completed_tokens}",
                        flush=True,
                    )


def write_summary_json(args: argparse.Namespace, total_tokens: int, total_bytes: int, ratio: float) -> None:
    if not args.summary_json:
        return
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(
            {
                "input_path": str(args.input_path),
                "output_path": str(args.output_path),
                "tokenizer": args.tokenizer.name,
                "vocab_path": str(args.tokenizer.vocab_path),
                "merges_path": str(args.tokenizer.merges_path),
                "dtype": "uint16",
                "tokens": total_tokens,
                "bytes": total_bytes,
                "bytes_per_token": ratio,
                "include_delimiter": args.include_delimiter,
                "workers": args.workers,
                "batch_bytes": args.batch_bytes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run_encode(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.tokenizer, args.special_token)
    validate_uint16_vocab(tokenizer)
    max_inflight_batches = args.max_inflight_batches or max(1, args.workers)

    if args.workers > 1:
        batch_stats = count_encoded_tokens_parallel(
            tokenizer_spec=args.tokenizer,
            input_path=args.input_path,
            special_token=args.special_token,
            include_delimiter=args.include_delimiter,
            max_docs=args.max_docs,
            max_bytes=args.max_bytes,
            batch_bytes=args.batch_bytes,
            workers=args.workers,
            max_inflight_batches=max_inflight_batches,
            progress_every_batches=args.progress_every_batches,
        )
        total_tokens = sum(stat.token_count for stat in batch_stats)
        total_bytes = sum(stat.byte_count for stat in batch_stats)
        if args.total_tokens is not None and args.total_tokens != total_tokens:
            raise ValueError(f"--total-tokens={args.total_tokens} but counted {total_tokens}")
        if args.total_bytes is not None and args.total_bytes != total_bytes:
            raise ValueError(f"--total-bytes={args.total_bytes} but counted {total_bytes}")
        ratio = total_bytes / total_tokens if total_tokens else float("nan")
        print(f"counted: bytes={total_bytes} tokens={total_tokens} bytes/token={ratio:.4f}")
        write_encoded_tokens_parallel(
            tokenizer_spec=args.tokenizer,
            input_path=args.input_path,
            output_path=args.output_path,
            special_token=args.special_token,
            include_delimiter=args.include_delimiter,
            max_docs=args.max_docs,
            max_bytes=args.max_bytes,
            batch_bytes=args.batch_bytes,
            workers=args.workers,
            max_inflight_batches=max_inflight_batches,
            progress_every_batches=args.progress_every_batches,
            batch_stats=batch_stats,
            total_tokens=total_tokens,
        )
        print(f"saved token ids to {args.output_path}")
        write_summary_json(args, total_tokens, total_bytes, ratio)
        return

    if args.total_tokens is not None:
        total_tokens = args.total_tokens
        total_bytes = args.total_bytes if args.total_bytes is not None else 0
    else:
        total_tokens, total_bytes = count_encoded_tokens(
            tokenizer=tokenizer,
            input_path=args.input_path,
            special_token=args.special_token,
            include_delimiter=args.include_delimiter,
            max_docs=args.max_docs,
            max_bytes=args.max_bytes,
            batch_bytes=args.batch_bytes,
            progress_every_batches=args.progress_every_batches,
        )
    ratio = total_bytes / total_tokens if total_tokens else float("nan")
    if args.total_tokens is not None:
        print(f"using provided counts: bytes={total_bytes} tokens={total_tokens} bytes/token={ratio:.4f}")
    else:
        print(f"counted: bytes={total_bytes} tokens={total_tokens} bytes/token={ratio:.4f}")

    write_encoded_tokens(
        tokenizer=tokenizer,
        input_path=args.input_path,
        output_path=args.output_path,
        special_token=args.special_token,
        include_delimiter=args.include_delimiter,
        max_docs=args.max_docs,
        max_bytes=args.max_bytes,
        total_tokens=total_tokens,
        batch_bytes=args.batch_bytes,
        progress_every_batches=args.progress_every_batches,
    )
    print(f"saved token ids to {args.output_path}")
    write_summary_json(args, total_tokens, total_bytes, ratio)


def repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def suite_tokenizer_spec(config: dict, key: str) -> TokenizerSpec:
    item = config["tokenizers"][key]
    return TokenizerSpec(
        name=item["name"],
        vocab_path=repo_path(item["vocab_path"]),
        merges_path=repo_path(item["merges_path"]),
    )


def require_existing(paths: list[Path]) -> None:
    missing = [rel_path(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))


def first_doc_boundary_batch(input_path: Path, special_token: str, target_bytes: int) -> tuple[str, int]:
    batches = iter_text_batches(
        input_path=input_path,
        special_token=special_token,
        include_delimiter=True,
        target_batch_bytes=target_bytes,
        max_docs=None,
        max_bytes=None,
    )
    try:
        return next(batches)
    except StopIteration as exc:
        raise ValueError(f"No text found in {rel_path(input_path)}") from exc


def run_suite_compression(config: dict, output_dir: Path) -> list[dict[str, object]]:
    special_token = config["special_token"]
    required = [repo_path(item["input_path"]) for item in config["compression"]]
    for item in config["tokenizers"].values():
        required.extend([repo_path(item["vocab_path"]), repo_path(item["merges_path"])])
    require_existing(required)

    rows: list[dict[str, object]] = []
    for item in config["compression"]:
        input_path = repo_path(item["input_path"])
        documents = sample_documents(
            input_path=input_path,
            special_token=special_token,
            sample_docs=int(config["sample_docs"]),
            seed=int(config["seed"]),
            max_docs=None,
            max_bytes=None,
        )
        if not documents:
            raise ValueError(f"No documents were sampled from {rel_path(input_path)}")

        for tokenizer_key in item["tokenizers"]:
            spec = suite_tokenizer_spec(config, tokenizer_key)
            tokenizer = load_tokenizer(spec, special_token)
            byte_count, token_count, ratio, _ = compute_ratio(
                tokenizer=tokenizer,
                documents=documents,
                special_token=special_token,
                include_delimiter=True,
            )
            row = {
                "dataset": item["name"],
                "input_path": rel_path(input_path),
                "tokenizer": spec.name,
                "sample_docs": len(documents),
                "seed": config["seed"],
                "bytes": byte_count,
                "tokens": token_count,
                "bytes_per_token": ratio,
            }
            rows.append(row)
            print(
                f"ratio {item['name']} / {spec.name}: "
                f"bytes={byte_count} tokens={token_count} bytes/token={ratio:.4f}",
                flush=True,
            )

    write_json(output_dir / "compression_ratio.json", rows)
    return rows


def run_suite_throughput(config: dict, output_dir: Path) -> list[dict[str, object]]:
    special_token = config["special_token"]
    target_bytes = int(config["throughput_target_bytes"])
    warmup = int(config["throughput_warmup"])
    repeats = int(config["throughput_repeats"])
    pile_bytes = int(config["pile_bytes"])
    required = [repo_path(item["input_path"]) for item in config["throughput"]]
    for item in config["tokenizers"].values():
        required.extend([repo_path(item["vocab_path"]), repo_path(item["merges_path"])])
    require_existing(required)

    rows: list[dict[str, object]] = []
    for item in config["throughput"]:
        input_path = repo_path(item["input_path"])
        spec = suite_tokenizer_spec(config, item["tokenizer"])
        tokenizer = load_tokenizer(spec, special_token)
        text, byte_count = first_doc_boundary_batch(input_path, special_token, target_bytes)

        token_count = 0
        for _ in range(warmup):
            token_count = len(tokenizer.encode(text))

        times: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            token_count = len(tokenizer.encode(text))
            times.append(time.perf_counter() - start)

        median_sec = statistics.median(times)
        bytes_per_sec = byte_count / median_sec
        row = {
            "name": item["name"],
            "input_path": rel_path(input_path),
            "tokenizer": spec.name,
            "target_bytes": target_bytes,
            "bytes": byte_count,
            "tokens": token_count,
            "warmup": warmup,
            "repeats": repeats,
            "times_sec": times,
            "median_sec": median_sec,
            "mb_per_sec": bytes_per_sec / 1_000_000,
            "pile_bytes": pile_bytes,
            "pile_serial_hours": pile_bytes / bytes_per_sec / 3600,
        }
        rows.append(row)
        print(
            f"throughput {item['name']}: bytes={byte_count} "
            f"median={median_sec:.4f}s speed={row['mb_per_sec']:.3f} MB/s",
            flush=True,
        )

    write_json(output_dir / "throughput.json", rows)
    return rows


def run_suite_encode_arrays(config: dict, output_dir: Path) -> None:
    encoding = config["encoding"]
    for item in config["encoded_arrays"]:
        spec = suite_tokenizer_spec(config, item["tokenizer"])
        summary_path = output_dir / f"{item['output_path'].replace('/', '_')}.summary.json"
        encode_args = argparse.Namespace(
            input_path=repo_path(item["input_path"]),
            special_token=config["special_token"],
            max_docs=None,
            max_bytes=None,
            include_delimiter=True,
            tokenizer=spec,
            output_path=repo_path(item["output_path"]),
            summary_json=summary_path,
            batch_bytes=int(encoding["batch_bytes"]),
            workers=int(encoding["workers"]),
            max_inflight_batches=int(encoding["max_inflight_batches"]),
            progress_every_batches=int(encoding["progress_every_batches"]),
            total_tokens=None,
            total_bytes=None,
        )
        print(f"encode {item['name']} -> {rel_path(encode_args.output_path)}", flush=True)
        run_encode(encode_args)


def run_suite_array_stats(config: dict, output_dir: Path, strict: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in config["encoded_arrays"]:
        input_path = repo_path(item["input_path"])
        output_path = repo_path(item["output_path"])
        row: dict[str, object] = {
            "name": item["name"],
            "input_path": rel_path(input_path),
            "output_path": rel_path(output_path),
            "tokenizer": config["tokenizers"][item["tokenizer"]]["name"],
        }
        if not output_path.exists():
            row["status"] = "missing_encoded_array"
            rows.append(row)
            if strict:
                raise FileNotFoundError(f"Missing encoded array: {rel_path(output_path)}")
            print(f"array-stats skip {item['name']}: missing {rel_path(output_path)}", flush=True)
            continue
        if not input_path.exists():
            row["status"] = "missing_raw_input"
            rows.append(row)
            if strict:
                raise FileNotFoundError(f"Missing raw input: {rel_path(input_path)}")
            print(f"array-stats skip {item['name']}: missing {rel_path(input_path)}", flush=True)
            continue

        token_ids = np.load(output_path, mmap_mode="r")
        byte_count = input_path.stat().st_size
        token_count = int(token_ids.shape[0])
        row.update(
            {
                "status": "ok",
                "dtype": str(token_ids.dtype),
                "tokens": token_count,
                "bytes": byte_count,
                "bytes_per_token": byte_count / token_count,
            }
        )
        rows.append(row)
        print(
            f"array-stats {item['name']}: tokens={token_count} "
            f"bytes/token={row['bytes_per_token']:.4f}",
            flush=True,
        )

    write_json(output_dir / "encoded_arrays.json", rows)
    return rows


def run_suite(args: argparse.Namespace) -> None:
    config_path = repo_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = repo_path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "config_path": rel_path(config_path),
        "output_dir": rel_path(output_dir),
        "compression": None,
        "throughput": None,
        "encoded_arrays": None,
    }
    if not args.skip_compression:
        summary["compression"] = run_suite_compression(config, output_dir)
    if not args.skip_throughput:
        summary["throughput"] = run_suite_throughput(config, output_dir)
    if args.encode_arrays:
        run_suite_encode_arrays(config, output_dir)
    if not args.skip_array_stats:
        summary["encoded_arrays"] = run_suite_array_stats(config, output_dir, args.strict)

    write_json(output_dir / "summary.json", summary)
    print(f"wrote {rel_path(output_dir / 'summary.json')}", flush=True)


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--special-token", default=DEFAULT_SPECIAL_TOKEN)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument(
        "--include-delimiter",
        action="store_true",
        help="Encode document delimiter special tokens when they are present in the data.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BPE tokenizer compression and dataset encoding experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ratio_parser = subparsers.add_parser("ratio", help="Sample documents and report bytes/token.")
    add_common_data_args(ratio_parser)
    ratio_parser.add_argument(
        "--tokenizer",
        type=parse_tokenizer_spec,
        action="append",
        required=True,
        help="Tokenizer spec as NAME:VOCAB_PATH:MERGES_PATH. Can be repeated.",
    )
    ratio_parser.add_argument("--sample-docs", type=int, default=10)
    ratio_parser.add_argument("--seed", type=int, default=0)
    ratio_parser.add_argument("--output-json", type=Path)
    ratio_parser.add_argument(
        "--include-token-ids",
        action="store_true",
        help="Include sampled document token id lists in --output-json.",
    )
    ratio_parser.set_defaults(func=run_ratio)

    encode_parser = subparsers.add_parser("encode", help="Encode a dataset to a uint16 .npy file.")
    add_common_data_args(encode_parser)
    encode_parser.add_argument(
        "--tokenizer",
        type=parse_tokenizer_spec,
        required=True,
        help="Tokenizer spec as NAME:VOCAB_PATH:MERGES_PATH.",
    )
    encode_parser.add_argument("--output-path", type=Path, required=True)
    encode_parser.add_argument("--summary-json", type=Path)
    encode_parser.add_argument("--batch-bytes", type=int, default=DEFAULT_BATCH_BYTES)
    encode_parser.add_argument("--workers", type=int, default=1)
    encode_parser.add_argument(
        "--max-inflight-batches",
        type=int,
        help="Maximum queued batches. Defaults to --workers to cap parent-process memory.",
    )
    encode_parser.add_argument("--progress-every-batches", type=int, default=10)
    encode_parser.add_argument(
        "--total-tokens",
        type=int,
        help="Skip the counting pass and use this token count to size the output .npy.",
    )
    encode_parser.add_argument(
        "--total-bytes",
        type=int,
        help="Optional byte count paired with --total-tokens for reporting bytes/token.",
    )
    encode_parser.set_defaults(func=run_encode)

    suite_parser = subparsers.add_parser(
        "suite",
        help="Run the README tokenizer experiments from configs/tokenizer_experiments.json.",
    )
    suite_parser.add_argument("--config", type=Path, default=DEFAULT_SUITE_CONFIG)
    suite_parser.add_argument("--output-dir", type=Path)
    suite_parser.add_argument("--skip-compression", action="store_true")
    suite_parser.add_argument("--skip-throughput", action="store_true")
    suite_parser.add_argument("--skip-array-stats", action="store_true")
    suite_parser.add_argument(
        "--encode-arrays",
        action="store_true",
        help="Generate full uint16 token-id arrays before computing array stats.",
    )
    suite_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if encoded arrays or raw inputs are missing during array stats.",
    )
    suite_parser.set_defaults(func=run_suite)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
