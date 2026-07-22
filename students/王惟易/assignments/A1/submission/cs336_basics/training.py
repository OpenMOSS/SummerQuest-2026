import torch
import numpy as np

from cs336_basics.model import cross_entropy

@torch.no_grad()
def clip_gradients(parameters, max_l2_norm: float) -> None:
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if len(gradients) == 0:
        return

    # 不能 torch.cat 所有梯度，否则会额外分配一个巨型 tensor
    total_squared_norm = sum([torch.sum(gradient**2) for gradient in gradients])

    total_norm = torch.sqrt(total_squared_norm)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for gradient in gradients:
            gradient.mul_(scale)

def get_batch(dataset, batch_size: int, context_length: int, device: torch.device | None=None) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(
        0, len(dataset) - context_length, size=batch_size
    )

    offsets = np.arange(context_length)
    indices = starts[:, None] + offsets[None, :]

    x_numpy = dataset[indices]
    y_numpy = dataset[indices + 1]

    x = torch.as_tensor(x_numpy, dtype=torch.long, device=device)
    y = torch.as_tensor(y_numpy, dtype=torch.long, device=device)

    return x, y

def save_checkpoint(model, optimizer, iteration: int, out) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }

    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer) -> int:
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]

def train_step(model, optimizer, x, y, max_l2_norm=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()
    if max_l2_norm is not None:
        clip_gradients(model.parameters(), max_l2_norm)
    optimizer.step()
    return loss.detach()

def evaluate(model, dataset, batch_size, context_length, device, num_batches) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            losses.append(loss)
    model.train(was_training)
    return torch.stack(losses).mean().item()
