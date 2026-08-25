import os

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    print("pid:", pid)
    print("offsets:", offsets)
    print("mask:", mask)

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous()

    output = torch.empty_like(x)
    n_elements = output.numel()
    block_size = 4
    grid = (triton.cdiv(n_elements, block_size),)

    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=block_size,
    )

    return output


def main() -> None:
    device = "cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda"
    x = torch.arange(10, device=device, dtype=torch.float32)
    y = torch.arange(10, 20, device=device, dtype=torch.float32)

    actual = triton_add(x, y)
    expected = x + y

    torch.testing.assert_close(actual, expected)
    print("x:", x)
    print("y:", y)
    print("result:", actual)


if __name__ == "__main__":
    main()
