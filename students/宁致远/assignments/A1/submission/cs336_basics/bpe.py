"""Byte-level BPE tokenizer training and encode/decode."""

from __future__ import annotations

import os
import regex as re
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO


def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Chunk a file into boundaries snapped to the next `split_special_token`."""
    assert isinstance(split_special_token, bytes)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    mini = 4096
    for bi in range(1, len(boundaries) - 1):
        pos = boundaries[bi]
        file.seek(pos)
        while True:
            chunk = file.read(mini)
            if not chunk:
                boundaries[bi] = file_size
                break
            at = chunk.find(split_special_token)
            if at != -1:
                boundaries[bi] = pos + at
                break
            pos += mini
    return sorted(set(boundaries))

# GPT-2 pretokenizer pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)


def _pretokenize_counts(text: str) -> dict[tuple[bytes, ...], int]:
    counts: dict[tuple[bytes, ...], int] = {}
    for m in _PAT_RE.finditer(text):
        b = m.group().encode("utf-8")
        key = tuple(bytes([c]) for c in b)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _process_chunk(args) -> dict[tuple[bytes, ...], int]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # keep special tokens out of pretoken counts so they never end up in merges
    if special_tokens:
        pattern = "|".join(re.escape(s) for s in special_tokens)
        pieces = re.split(pattern, chunk)
    else:
        pieces = [chunk]
    total: dict[tuple[bytes, ...], int] = {}
    for piece in pieces:
        for k, v in _pretokenize_counts(piece).items():
            total[k] = total.get(k, 0) + v
    return total


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 4,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE. Returns (vocab id->bytes, merges list)."""
    # base vocab = 256 bytes then specials
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, tok in enumerate(special_tokens):
        vocab[256 + i] = tok.encode("utf-8")
    next_id = 256 + len(special_tokens)

    # parallel pretokenize over file chunks
    split_bytes = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_bytes)
    tasks = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    counts: dict[tuple[bytes, ...], int] = {}
    if num_processes > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as ex:
            for partial in ex.map(_process_chunk, tasks):
                for k, v in partial.items():
                    counts[k] = counts.get(k, 0) + v
    else:
        for t in tasks:
            for k, v in _process_chunk(t).items():
                counts[k] = counts.get(k, 0) + v

    # words as token-bytes lists + reverse indexes: pair -> count, pair -> word ids
    words: list[list[bytes]] = [list(k) for k in counts]
    freqs: list[int] = list(counts.values())
    pair_count: dict[tuple[bytes, bytes], int] = {}
    pair_where: dict[tuple[bytes, bytes], set[int]] = {}
    for i, w in enumerate(words):
        f = freqs[i]
        for a, b in zip(w, w[1:]):
            p = (a, b)
            pair_count[p] = pair_count.get(p, 0) + f
            pair_where.setdefault(p, set()).add(i)

    import heapq
    # max-heap keyed (-count, inverted pair) → tie-break lex-max
    class _Neg:
        __slots__ = ("p",)
        def __init__(self, p): self.p = p
        def __lt__(self, o): return self.p > o.p
    heap = [(-c, _Neg(p), p) for p, c in pair_count.items()]
    heapq.heapify(heap)

    def _push(p, c):
        heapq.heappush(heap, (-c, _Neg(p), p))

    merges: list[tuple[bytes, bytes]] = []
    target_merges = vocab_size - len(vocab)
    while len(merges) < target_merges and heap:
        # skip stale entries (lazy deletion)
        while heap:
            neg_c, _, p = heap[0]
            cur = pair_count.get(p, 0)
            if cur == -neg_c and cur > 0:
                best = p
                break
            heapq.heappop(heap)
        else:
            break
        if pair_count.get(best, 0) <= 0:
            break
        merges.append(best)
        vocab[next_id] = best[0] + best[1]
        next_id += 1

        a, b = best
        merged = a + b
        affected = list(pair_where.get(best, ()))
        touched: dict[tuple[bytes, bytes], None] = {}  # pairs to re-push once
        for i in affected:
            w = words[i]
            f = freqs[i]
            new_w: list[bytes] = []
            j = 0
            for p in zip(w, w[1:]):
                pair_count[p] = pair_count.get(p, 0) - f
                touched[p] = None
                if pair_count[p] <= 0:
                    pair_count.pop(p, None)
                s = pair_where.get(p)
                if s is not None:
                    s.discard(i)
                    if not s:
                        pair_where.pop(p, None)
            while j < len(w):
                if j < len(w) - 1 and w[j] == a and w[j + 1] == b:
                    new_w.append(merged)
                    j += 2
                else:
                    new_w.append(w[j])
                    j += 1
            words[i] = new_w
            for p in zip(new_w, new_w[1:]):
                pair_count[p] = pair_count.get(p, 0) + f
                pair_where.setdefault(p, set()).add(i)
                touched[p] = None
        for p in touched:
            c = pair_count.get(p, 0)
            if c > 0:
                _push(p, c)

    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)
        specials = list(special_tokens) if special_tokens else []
        self.special_tokens = specials
        self._bytes_to_id = {v: k for k, v in self.vocab.items()}
        # append specials that aren't already in vocab
        for s in specials:
            sb = s.encode("utf-8")
            if sb not in self._bytes_to_id:
                nid = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[nid] = sb
                self._bytes_to_id[sb] = nid
        self.merges = list(merges)
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        # longer specials first, else "<|eot|><|eot|>" would be split by "<|eot|>"
        if specials:
            ordered = sorted(specials, key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(s) for s in ordered) + ")")
        else:
            self._special_re = None

    @classmethod
    def from_files(
        cls,
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Load GPT-2 style vocab.json + merges.txt (byte-unicode mapping)."""
        import json

        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        byte_decoder = {chr(c): b for b, c in zip(bs, cs)}
        with open(vocab_path) as f:
            raw = json.load(f)
        vocab = {i: bytes([byte_decoder[c] for c in tok]) for tok, i in raw.items()}
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_path) as f:
            for line in f:
                line = line.rstrip()
                if not line or len(line.split(" ")) != 2:
                    continue
                a, b = line.split(" ")
                merges.append((bytes([byte_decoder[c] for c in a]), bytes([byte_decoder[c] for c in b])))
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_piece(self, text: str) -> list[int]:
        """Apply merges greedily by lowest rank on one pretoken piece."""
        ids: list[int] = []
        for m in _PAT_RE.finditer(text):
            b = m.group().encode("utf-8")
            parts = [bytes([c]) for c in b]
            while len(parts) > 1:
                best_i = -1
                best_rank = None
                for i in range(len(parts) - 1):
                    r = self._rank.get((parts[i], parts[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank = r
                        best_i = i
                if best_i < 0:
                    break
                parts[best_i : best_i + 2] = [parts[best_i] + parts[best_i + 1]]
            for p in parts:
                ids.append(self._bytes_to_id[p])
        return ids

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        if self._special_re is None:
            return self._bpe_encode_piece(text)
        out: list[int] = []
        for piece in self._special_re.split(text):
            if not piece:
                continue
            if piece in self.special_tokens:
                out.append(self._bytes_to_id[piece.encode("utf-8")])
            else:
                out.extend(self._bpe_encode_piece(piece))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Stream-encode: flush at newline so memory stays O(one line)."""
        buf = ""
        for chunk in iterable:
            buf += chunk
            if "\n" in buf:
                last = buf.rfind("\n") + 1
                head, buf = buf[:last], buf[last:]
                for i in self.encode(head):
                    yield i
        if buf:
            for i in self.encode(buf):
                yield i

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")
