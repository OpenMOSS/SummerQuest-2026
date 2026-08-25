import copy

import torch
from torch import nn

from cs336_systems.a2k.checkpointing import run_checkpointed_layers


def build_layers() -> nn.ModuleList:
    return nn.ModuleList(
        [
            nn.Sequential(
                nn.Linear(4, 4),
                nn.SiLU(),
            )
            for _ in range(5)
        ]
    )


def main() -> None:
    torch.manual_seed(0)

    baseline_layers = build_layers()
    checkpointed_layers = copy.deepcopy(baseline_layers)

    baseline_input = torch.randn(2, 3, 4, requires_grad=True)
    checkpointed_input = baseline_input.detach().clone().requires_grad_(True)

    baseline_output = run_checkpointed_layers(
        baseline_layers,
        baseline_input,
        checkpoint_block_size=None,
    )
    baseline_output.square().mean().backward()

    checkpointed_output = run_checkpointed_layers(
        checkpointed_layers,
        checkpointed_input,
        checkpoint_block_size=2,
    )
    checkpointed_output.square().mean().backward()

    torch.testing.assert_close(checkpointed_output, baseline_output)
    torch.testing.assert_close(checkpointed_input.grad, baseline_input.grad)

    for (
        (baseline_name, baseline_parameter),
        (checkpointed_name, checkpointed_parameter),
    ) in zip(
        baseline_layers.named_parameters(),
        checkpointed_layers.named_parameters(),
        strict=True,
    ):
        assert baseline_name == checkpointed_name
        torch.testing.assert_close(checkpointed_parameter.grad, baseline_parameter.grad)

    print("output and all gradients match")


if __name__ == "__main__":
    main()
