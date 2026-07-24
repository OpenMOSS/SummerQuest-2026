"""unicode1/unicode2 answers + AdamW memory/FLOPs numbers.

Run this file to get the printable outputs used by the report.
"""

from __future__ import annotations

import textwrap


def unicode1():
    print("== unicode1 ==")
    # Q(a): what does chr(0) return?
    c = chr(0)
    print("chr(0) =", repr(c))
    print("its __repr__ =", c.__repr__())
    # Q(b): what happens when you print(chr(0))?
    print("print(chr(0)) prints:", repr(str(chr(0))))
    # Q(c): what happens when it appears inside another string?
    s = "hello" + chr(0) + "world"
    print("repr:", repr(s))
    print("print output:", s)
    print()


def unicode2():
    print("== unicode2 ==")
    # Q(a): why byte-level BPE?
    print("byte-level BPE reasoning:")
    print(textwrap.dedent(
        """
        - Every byte value 0..255 is a valid initial token, so any UTF-8 text can be
          represented and no OOV appears.
        - Unicode-level BPE would need a codepoint-level base of ~1.1M and still miss
          new codepoints from the wild web; sequences of surrogates/CJK/emoji would blow
          up the initial vocabulary.
        - Byte-level keeps the base at 256 while the merge process still ends up
          learning common Unicode substrings (e.g. UTF-8 sequences for common
          multi-byte chars become single tokens).
        """
    ).strip())
    print()
    # Q(b): show a mis-decoded byte
    b = bytes([0xC3])           # incomplete UTF-8 lead byte
    print("bytes([0xC3]).decode('utf-8', errors='replace') =",
          repr(b.decode("utf-8", errors="replace")))
    b2 = bytes([0xC3, 0xA9])    # é
    print("bytes([0xC3,0xA9]).decode('utf-8') =", repr(b2.decode("utf-8")))
    print("=> byte-level tokenizer never splits inside a multibyte codepoint after"
          " merges are applied, because the byte-pair merges learn 0xC3 0xA9 together.")
    print()


def adamw_memory():
    print("== AdamW memory ==")
    print(textwrap.dedent(
        """
        Per parameter, AdamW keeps:
          - master parameters (fp32)      : 4 bytes
          - first-moment m_t   (fp32)     : 4 bytes
          - second-moment v_t  (fp32)     : 4 bytes
          - live parameter copy (bf16)    : 2 bytes
          - gradient buffer     (bf16)    : 2 bytes
        Total ~= 16 bytes / param.

        For a p-parameter model:
          weights+grad+optim ≈ 16 * p bytes.

        Example (from scripts/gpt2_flops.py):
          - gpt2-small  (124M)  ->  ~2.0 GB
          - gpt2-medium (355M)  ->  ~5.7 GB
          - gpt2-large  (774M)  -> ~12.4 GB
          - gpt2-xl     (1.6B)  -> ~24.9 GB
        """
    ).strip())
    print()


def adamw_flops():
    print("== AdamW FLOPs / step ==")
    print(textwrap.dedent(
        """
        Per step, AdamW updates for each parameter roughly:
          m  = beta1*m + (1-beta1)*g              # 3 fma ops (~6 flops)
          v  = beta2*v + (1-beta2)*g*g            # 4 fma ops (~8 flops)
          m_hat = m / (1-beta1**t)
          v_hat = v / (1-beta2**t)
          p -= lr * m_hat / (sqrt(v_hat) + eps)   # 4 ops (sqrt+add+div+mul)
          p -= lr * wd * p                        # 2 ops
        ≈ 20 flops / parameter, i.e. ~5x an SGD step.

        Compared with a forward+backward pass at Ftok · L · B flops,
        AdamW is essentially free (a few % of a step at N ≥ 100M scale).

        Example: gpt2-small (124M params) -> 2.5 GFLOPs / step optimizer work,
        vs ~300 GFLOPs / seq forward + ~600 GFLOPs backward.
        """
    ).strip())
    print()


def training_time_estimate():
    print("== Training time estimate ==")
    print(textwrap.dedent(
        """
        Total FLOPs for training ≈ 6 * N * T   where
          N = number of parameters, T = tokens processed
        (this is the standard "6ND" scaling law approximation).

        Wall-clock ≈ Total FLOPs / (device throughput * MFU).

        Our TinyStories baseline (~30M params, 128*256*10_000 ≈ 3.3e8 tokens):
          Total ≈ 6 * 3e7 * 3.3e8 ≈ 6e16 FLOPs = 60 PFLOPs
          On a single A100 at ~150 TFLOPs bf16 * 0.35 MFU ≈ 50 TFLOPs:
            wall ≈ 60e15 / 50e12 ≈ 1200 s ≈ 20 min.
        """
    ).strip())
    print()


if __name__ == "__main__":
    unicode1()
    unicode2()
    adamw_memory()
    adamw_flops()
    training_time_estimate()
