import typing
import os

import torch
import torch.nn as nn

def save_checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str|os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    dict = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(dict, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model:nn.Module, optimizer:torch.optim.Optimizer):
    dict = torch.load(src, map_location=lambda storage, loc: storage)
    model.load_state_dict(dict["model_state"])
    optimizer.load_state_dict(dict["optimizer_state"])
    return dict["iteration"]