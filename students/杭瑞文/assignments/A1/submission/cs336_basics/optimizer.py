import torch
import math
from collections.abc import Callable, Iterable

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_stable = logits - max_logits
    
    sum_exp = torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True)
    log_probs = logits_stable - torch.log(sum_exp)
    targets = targets.unsqueeze(-1)
    nll = -torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
    return nll.mean()
    

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: tuple[float, float] = (0.9, 0.999), 
        eps: float = 1e-8, 
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            lambda_ = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                p.data.mul_(1.0 - lr * lambda_)

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

        return loss

def get_lr_cosine_schedule(
    t: int, 
    alpha_max: float, 
    alpha_min: float, 
    T_w: int, 
    T_c: int
) -> float:
    if t < T_w:
        if T_w == 0:
            return alpha_max
        return (t / T_w) * alpha_max

    elif T_w <= t <= T_c:
        if T_c == T_w:
            return alpha_min
        
        progress = (t - T_w) / (T_c - T_w)
        cosine_decay = 0.5 * (1.0 + math.cos(progress * math.pi))
        
        return alpha_min + cosine_decay * (alpha_max - alpha_min)

    else:
        return alpha_min
    
def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6) -> float:
    parameters = list(parameters)
    grads = [p.grad.detach() for p in parameters if p.grad is not None]

    if not grads:
        return 0.0

    # Accumulating squared norms avoids allocating one model-sized flattened tensor.
    norm_device = grads[0].device
    squared_norm = torch.zeros((), device=norm_device, dtype=torch.float32)
    for grad in grads:
        squared_norm += torch.sum(grad.detach().float() ** 2)
    total_norm_tensor = torch.sqrt(squared_norm)
    total_norm = total_norm_tensor.item()

    if not math.isfinite(total_norm):
        return total_norm

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)

    return total_norm
