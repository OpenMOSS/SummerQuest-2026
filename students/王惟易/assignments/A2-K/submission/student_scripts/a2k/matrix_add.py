import os

import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)

    row_offsets = pid_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    col_offsets = pid_col * BLOCK_COLS + tl.arange(0, BLOCK_COLS)

    linear_offsets = row_offsets[:, None] * n_cols + col_offsets[None, :]

    mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)

    print("pid_row: ", pid_row)
    print("pid_col: ", pid_col)
    print("row_offsets: ", row_offsets)
    print("row_offsets[:, None]:\n", row_offsets[:, None])
    print("col_offsets: ", col_offsets)
    print("col_offsets[None, :]:\n", col_offsets[None, :])
    print("linear_offsets:\n", linear_offsets)
    print("mask:\n", mask)
    print("-" * 50)

    x = tl.load(x_ptr + linear_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + linear_offsets, mask=mask, other=0.0)

    tl.store(output_ptr + linear_offsets, x + y, mask=mask)


def triton_matrix_add(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.ndim == 2
    assert x.is_contiguous() and y.is_contiguous()

    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    block_rows = 2
    block_cols = 4
    grid = (
        triton.cdiv(n_rows, block_rows),
        triton.cdiv(n_cols, block_cols),
    )

    matrix_add_kernel[grid](
        x,
        y,
        output,
        n_rows,
        n_cols,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
    )

    print("grid: ", grid)

    return output


def main() -> None:
    interpreting = os.environ.get("TRITON_INTERPRET") == "1"
    device = "cpu" if interpreting else "cuda"

    x = torch.arange(
        5 * 7,
        device=device,
        dtype=torch.float32,
    ).reshape(5, 7)
    y = torch.full_like(x, 100)

    actual = triton_matrix_add(x, y)
    expected = x + y

    torch.testing.assert_close(actual, expected)
    print("x:\n", x)
    print("result:\n", actual)


if __name__ == "__main__":
    main()
