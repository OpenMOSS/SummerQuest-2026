import torch
import numpy as np
from cs336_basics.model import TransformerLM
from cs336_basics.training import cross_entropy, AdamW, get_batch

def main():
    device = "cpu"
    vocab_size = 10000

    # ── 小模型(快)──
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3)

    # ── 固定一个 batch,反复训练它 ──
    train_data = np.load("data/ts_train.npy", mmap_mode="r")
    x, y = get_batch(train_data, batch_size=4, context_length=32, device=device)

    for step in range(300):
        logits = model(x)                                    # (4, 32, vocab)
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step={step:3d}  loss={loss.item():.4f}")

if __name__ == "__main__":
    main()