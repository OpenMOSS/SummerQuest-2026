import torch


def main() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    values = torch.tensor([10, 20, 30, 40], dtype=torch.float32)

    row_max = torch.tensor(float("-inf"))
    row_sum = torch.tensor(0.0)
    output_accumulator = torch.tensor(0.0)

    tile_size = 2

    tile_scores = scores.split(tile_size)
    tile_values = values.split(tile_size)

    for tile_score, tile_value in zip(tile_scores, tile_values, strict=True):
        # process each tile
        tile_max = tile_score.max()
        tile_p = torch.exp(tile_score - tile_max)
        tile_sum = tile_p.sum()
        tile_output = (tile_p * tile_value).sum()

        new_max = torch.maximum(row_max, tile_max)
        old_correction = torch.exp(row_max - new_max)
        tile_correction = torch.exp(tile_max - new_max)

        new_sum = old_correction * row_sum + tile_correction * tile_sum
        new_output = old_correction * output_accumulator + tile_correction * tile_output

        row_max, row_sum, output_accumulator = new_max, new_sum, new_output
        print(f"m={row_max.item():.6f}, l={row_sum.item():.6f}, o={output_accumulator.item():.6f}")

    online_output = output_accumulator / row_sum
    online_lse = row_max + torch.log(row_sum)

    expected_output = torch.softmax(scores, dim=0) @ values
    expected_lse = torch.logsumexp(scores, dim=0)

    torch.testing.assert_close(online_output, expected_output)
    torch.testing.assert_close(online_lse, expected_lse)


if __name__ == "__main__":
    main()
