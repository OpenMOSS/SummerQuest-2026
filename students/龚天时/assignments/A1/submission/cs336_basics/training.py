import torch
import math
import numpy as np

def cross_entropy(logits, targets) -> torch.Tensor:
    logits = logits - logits.max(dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))
    correct = logits[torch.arange(logits.shape[0]), targets]
    return (-correct + log_sum_exp).mean()

import torch, math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]                    # α
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]          # λ
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]


                lr_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * wd * p.data
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                p.data -= lr_t * m / (v.sqrt() + eps)

        return loss
    
def get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr
    ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)  
            
def get_batch(dataset, batch_size, context_length, device):
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    
    x = np.stack([dataset[s : s+context_length] for s in starts])
    y = np.stack([dataset[s+1 : s+context_length+1] for s in starts])
    
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),          
        "optimizer": optimizer.state_dict(), 
        "iteration": iteration,  
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(src, model, optimizer):
    device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]