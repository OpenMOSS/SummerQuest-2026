from time import perf_counter

import torch

from cs336_systems.a2k.attention import explicit_attention


def timed_call(function, *args):
    start = perf_counter()
    output = function(*args)
    elapsed_ms = (perf_counter() - start) * 1000
    return output, elapsed_ms


def main() -> None:
    torch.manual_seed(0)

    q = torch.randn(2, 128, 64)
    k = torch.randn(2, 128, 64)
    v = torch.randn(2, 128, 64)

    eager_output = explicit_attention(q, k, v, True)

    compiled_attention = torch.compile(explicit_attention, fullgraph=True)

    cold_output, cold_ms = timed_call(
        compiled_attention,
        q,
        k,
        v,
        True,
    )

    cached_output, cached_ms = timed_call(
        compiled_attention,
        q,
        k,
        v,
        True,
    )

    torch.testing.assert_close(cold_output, eager_output)
    torch.testing.assert_close(cached_output, eager_output)

    print(f"cold_ms: {cold_ms:.2f} ms")
    print(f"cached_ms: {cached_ms:.2f} ms")


if __name__ == "__main__":
    main()
