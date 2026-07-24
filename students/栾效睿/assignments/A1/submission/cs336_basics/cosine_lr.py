import math


def cosine_lr(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:

    assert t_c > t_w and lr_min < lr_max
    if t < t_w:
        return (t / t_w) * lr_max
    elif t <= t_c:
        return (
            lr_min
            + (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min) / 2
        )
    else:
        return lr_min
