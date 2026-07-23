import math

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
        
    # 2. 后退火阶段 (Post-annealing): 保持最小学习率
    if it > cosine_cycle_iters:
        return min_learning_rate
        
    # 3. 余弦退火阶段 (Cosine annealing): 平滑下降
    # 计算当前处于退火阶段的比例 (0.0 到 1.0 之间)
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    
    # 根据余弦公式计算系数 (1.0 到 0.0 之间)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    
    # 计算当前学习率
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)