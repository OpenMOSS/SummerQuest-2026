# A1 公开提交：张境航


## 实验环境

- Python: 3.13.14
- PyTorch: ...
- Device: ...
- Tests:
  47 passed, 1 xfailed

## 实现内容

完成：
- Transformer 基础组件
- Multi-head Attention
- RoPE
- RMSNorm
- AdamW
- BPE tokenizer
...

## 测试结果

pytest:

47 passed, 1 xfailed

## 实验记录

...

## RoPE 修正说明

初版实现中，RoPE 对 batch token positions 的广播处理存在不足。

当 q/k 输入形状为：

(batch, num_heads, sequence_length, head_dim)

而 token_positions 为：

(batch, sequence_length)

时，cos/sin cache 索引后的结果无法正确广播到 attention head 维度。

当前已修复：

- 在应用 RoPE 前增加 singleton head dimension；
- 支持 batch token_positions 自动广播。

相关测试：

uv run pytest tests/test_model.py::test_rope -v
uv run pytest tests/test_model.py::test_multihead_self_attention_with_rope -v

均通过。


## 关于训练实验部分说明

当前 A1 Basics 提交主要完成 Transformer 基础组件、优化器、tokenizer 和训练辅助模块。

完整训练实验需要额外的：
- GPT training pipeline
- OpenWebText 数据处理流程
- GPU 长时间训练环境
- 实验配置文件

当前公开提交环境未包含上述训练流程，因此未提交未经实际运行验证的实验数据。

后续将在完整训练环境下进一步开展相关实验。