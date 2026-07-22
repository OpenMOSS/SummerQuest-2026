import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
from .model import (
    Embedding,
    Linear,
    MultiHeadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SiLUFeedForward,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from .tokenizer import Tokenizer, save_tokenizer, train_bpe
from .training import AdamW, clip_gradients, cosine_schedule, cross_entropy, get_batch, load_checkpoint, save_checkpoint
