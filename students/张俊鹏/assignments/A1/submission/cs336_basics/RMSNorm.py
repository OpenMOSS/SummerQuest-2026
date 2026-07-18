import torch
import torch.nn as nn

"""
归一化操作：保证数值稳定性
"""

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        构造 RMSNorm 模块。
        """
        super().__init__()
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.weight = nn.Parameter(torch.empty(d_model, **factory_kwargs))
        
        # 按要求初始化
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理形状为 (batch_size, sequence_length, d_model) 的输入张量。
        """
        # 记录原始的数据类型，以便后续还原
        orig_dtype = x.dtype
        
        # 强制上采样 (Upcast) 到 float32，防止计算平方时数值溢出
        x_f32 = x.to(torch.float32)
        
        # 计算均方根 (Root Mean Square) 的分母部分
        # 公式: variance = mean(x^2)
        # 注意：必须 keepdim=True，以保证算出的方差形状为 (..., 1)，从而能与 x 进行广播除法
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        
        # 计算归一化结果
        # 使用 torch.rsqrt (倒数平方根) 比 1 / torch.sqrt() 在底层 C++ 实现上更稳定、更高效
        x_normed = x_f32 * torch.rsqrt(variance + self.eps)
        
        # 下采样 (Downcast) 回原始的数据类型 (比如 bfloat16/float16)
        x_normed = x_normed.to(orig_dtype)
        
        # x_normed 形状: (Batch, Seq_Len, d_model)
        # weight 形状:   (d_model,)
        # PyTorch 会自动进行广播 (Broadcasting) 相乘

        return x_normed * self.weight
        # return x # 删除 RMSNorm