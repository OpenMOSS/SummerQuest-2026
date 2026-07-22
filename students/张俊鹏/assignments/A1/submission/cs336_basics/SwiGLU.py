import torch
import torch.nn as nn

"""
激活函数
"""

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 1. 按照参数量对齐原则计算 d_ff: 8/3 * d_model
        self.d_ff = d_ff
        
        # 3. 实例化三个线性层
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, **factory_kwargs)  # Gate 投影 (d_model -> d_ff)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, **factory_kwargs)  # Up 投影 (d_model -> d_ff)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, **factory_kwargs)  # Down 投影 (d_ff -> d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 ( SiLU(xW1) ⊗ xW3 ) W2
        """
        # 1. 经过门控网络和向上投影网络
        gate = self.w1(x)
        up = self.w3(x)
        
        # 2. 手动实现 SiLU 以保证数值稳定性: x * sigmoid(x)
        silu_gate = gate * torch.sigmoid(gate)
        
        # 3. 元素级相乘 (Hadamard Product)
        hidden = silu_gate * up
        
        # 4. 投影回原始的 d_model 维度
        return self.w2(hidden)
