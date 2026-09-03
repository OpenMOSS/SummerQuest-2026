import torch
import math
from torch.optim import Optimizer

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    # 将 logits 展平为 (N, vocab_size)，targets 展平为 (N,)
    vocab_size = logits.size(-1)
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    logits_max = torch.max(logits_flat, dim=-1, keepdim=True).values
    logits_shifted = logits_flat - logits_max
    
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True))
    
    log_probs = logits_shifted - log_sum_exp
    
    nll = -log_probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
    
    return nll.mean()

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr / bias_correction1

                # 对二阶矩进行偏差校正后的 sqrt
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                p.mul_(1 - lr * weight_decay)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """
    计算带线性 warmup 的余弦退火学习率。

    如果 it < warmup_iters:
        lr = max_lr * (it / warmup_iters)
    否则:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_lr + 0.5 * (1 + cos(pi * progress)) * (max_lr - min_lr)
    """
    
    if it < warmup_iters:
        # 线性 warmup 阶段
        return max_learning_rate * (it /warmup_iters)
    else:
        # 余弦衰减阶段
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        progress = min(1.0, max(0.0, progress))
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    
def clip_gradients(
    parameters,
    max_12_norm: float,
    eps: float = 1e-6
):
    """
    对所有参数的梯度进行全局 L2 范数裁剪（原地修改 p.grad）。

    计算 total_norm = sqrt( sum( ||grad||^2 ) )
    如果 total_norm > max_l2_norm:
        缩放系数 = max_l2_norm / (total_norm + eps)
        对每个梯度乘以该系数
    """
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    
    total_norm = 0.0
    for p in params_with_grad:
        grad = p.grad
        total_norm += grad.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)
    
    clip_coef = max_12_norm / (total_norm + eps)
    if clip_coef < 1.0:
        for p in params_with_grad:
            p.grad.mul_(clip_coef)