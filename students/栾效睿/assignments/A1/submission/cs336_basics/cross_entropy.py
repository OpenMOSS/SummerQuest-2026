import torch
import torch.nn as nn


def cross_entropy(output: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    assert output.dim() == 2 and seq.dim() == 1
    # output(batch_size, vocab_size).  seq(batch_size)
    batch_size = output.shape[0]
    # 每个位置的分数最大值
    x_max = output.max(dim=-1, keepdim=True).values

    # substract max_vlaue
    x_stable = output - x_max

    # the_target logits
    x_target = x_stable.gather(dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)

    # 求分母
    exp_sum = torch.exp(x_stable).sum(dim=-1)

    # print("x_target:", x_target.shape)
    # print("exp_sum:", exp_sum.shape)

    res = torch.log(exp_sum).sum() - x_target.sum()

    return res / batch_size
