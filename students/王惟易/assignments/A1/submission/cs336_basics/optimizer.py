import torch
import math

class AdamW(torch.optim.Optimizer):
    """
    优化器的工作本质是在 autograd 之外维护并原地改写长期状态。
    """
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    # 参数没有参加本次反向传播，完全跳过，不做 weight decay
                    # 但如果 parameter.grad == 0，那么仍会正常执行 weight decay，不会跳过
                    continue

                gradient = parameter.grad
                state = self.state[parameter]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)

                m = state["m"]
                v = state["v"]

                m.mul_(beta1) # 原地操作, `m = beta1 * m`
                m.add_(gradient, alpha=1-beta1) # `m = beta1*m + (1-beta1)*gradient`

                v.mul_(beta2)
                v.addcmul_(gradient, gradient, value=1-beta2) # `v=beta2*v+(1-beta2)*gradient^2`

                state["t"] += 1

                t = state["t"]
                lr = group["lr"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                adjusted_lr = (
                    lr * math.sqrt(1 - beta2**t) / (1-beta1**t)
                )

                denominator = v.sqrt().add_(eps) # v.sqrt() 会创建新 tensor，v.sqrt_() 会直接修改 v

                parameter.mul_(1 - lr*weight_decay) # `parameter = parameter - lr*weight_decay*parameter`
                parameter.addcdiv_(
                    m,
                    denominator,
                    value=-adjusted_lr
                ) # `parameter += value * m / denominator`

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

        lr = min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    else:
        lr = min_learning_rate

    return lr

if __name__ == "__main__":
    pass