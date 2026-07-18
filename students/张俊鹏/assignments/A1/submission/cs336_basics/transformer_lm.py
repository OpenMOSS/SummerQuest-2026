import torch
import torch.nn as nn
from cs336_basics.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        
        # 1. 词嵌入层 (Token Embedding)
        # 维度: (vocab_size, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 多层 Transformer Blocks
        # 这里直接复用你之前写好且测试通过的 TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, 
                num_heads=num_heads, 
                d_ff=d_ff, 
                max_seq_len=context_length, 
                theta=theta
            ) for _ in range(num_layers)
        ])
        
        # 3. 最终的层归一化 (Final Layer Norm)
        # 注意：如果你的架构里用的是 RMSNorm，这里请换成你的 RMSNorm 类
        self.final_norm = nn.RMSNorm(d_model) 
        
        # 4. 语言模型头 (LM Head)
        # 将 Transformer 提取出的特征映射回词汇表，通常不使用偏置项 (bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
            input_ids: 形状为 (batch_size, seq_len) 的整数张量，代表输入的词索引。
        返回:
            logits: 形状为 (batch_size, seq_len, vocab_size) 的张量，未归一化的概率分布。
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 动态生成位置索引，并确保与前向传播的 batch 维度匹配
        # 这也是你之前 debug 成功的关键：生成 (batch_size, seq_len) 的 positions
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 1. 获取嵌入表示 -> (batch_size, seq_len, d_model)
        x = self.token_embedding(input_ids)
        
        # 2. 依次穿过所有的 Transformer Block
        for block in self.blocks:
            # 记得要把生成的位置索引传给每一个 block
            x = block(x, token_positions=positions)
            
        # 3. 经过最后的归一化层
        x = self.final_norm(x)
        
        # 4. 经过输出投影头，得到词表 logits -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits