"""A2-K 可提交实现包。

此目录是课程同步脚本允许复制的代码边界。FlashAttention 实现放在这里，
顶层兼容模块仍保留给已有实验日志和旧命令使用。
"""

from .flash_attention import PyTorchFlashAttention, TritonFlashAttention

__all__ = ["PyTorchFlashAttention", "TritonFlashAttention"]
