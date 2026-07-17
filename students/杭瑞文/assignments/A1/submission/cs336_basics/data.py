import numpy as np
import torch
import os
from typing import BinaryIO, IO, Union

def get_batch(
    data: np.ndarray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = len(data) - context_length
    
    ix = np.random.randint(0, max_idx, size=batch_size)
    
    x_list = [data[i : i + context_length] for i in ix]
    y_list = [data[i + 1 : i + context_length + 1] for i in ix]
    
    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)
    
    return x, y

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
) -> None:
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    
    torch.save(checkpoint_dict, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    
    checkpoint_dict = torch.load(src, map_location="cpu")    
    model.load_state_dict(checkpoint_dict["model_state_dict"])    
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])    
    return checkpoint_dict["iteration"]