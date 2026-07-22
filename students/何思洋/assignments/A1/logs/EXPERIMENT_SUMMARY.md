# Assignment 1 Experiment Summary

## Text Generation

Generation script:

```sh
conda run -n a1-basics uv run --no-sync python scripts/generate.py ...
```

The script loads a saved `checkpoint["model"]`, reconstructs the assignment TransformerLM, encodes the prompt with the saved BPE tokenizer, and samples autoregressively with temperature and nucleus sampling.

### TinyStories Baseline

- Checkpoint: `artifacts/runs/tinystories_baseline/checkpoints/tinystories_baseline_step10000.pt`
- Tokenizer: `artifacts/tokenizers/tinystories_train_10k.json`
- Prompt: `Once upon a time, there was a little girl named Lily who found a magic key.`
- Sampling: temperature 0.8, top-p 0.9, seed 42
- Generated tokens: 123, stopped at `<|endoftext|>`

Sample:

```text
Once upon a time, there was a little girl named Lily who found a magic key. The key could unlock a secret door in her house. Lily was very excited and wanted to see what was behind the door.
Lily unlocked the secret door and found a room full of toys and treats. She played and ate treats all day long. But then, something unexpected happened. The toys started to talk! They said, "Thank you for unlocking the secret door!"
Lily was very surprised and happy. She played with the toys and shared them with her friends. They all had a fun day in the secret room. And from that day on, Lily and her toys lived happily ever after.
<|endoftext|>
```

Brief assessment: the sample is coherent, child-story-like, and closes naturally. It is repetitive in structure but matches the TinyStories domain well.

### OWT 512MB Full32K

- Checkpoint: `artifacts/runs/owt_512M_full32k/checkpoints/owt_512M_full32k_step10000.pt`
- Tokenizer: `artifacts/tokenizers/owt_train_full_32k.json`
- Prompt: `The future of artificial intelligence depends on`
- Sampling: temperature 0.8, top-p 0.9, seed 42
- Generated tokens: 256

Sample:

```text
The future of artificial intelligence depends on the fact that the human condition, for example, is closely related to the biological processes involved in processing human physiological processes and social processes, and therefore the way we do not know. If we are going to do some kind of that, then we can do some kind of research.

The human condition, however, is not a single instance of human activity. However, the human condition may not be an acceptable part of a human condition, and it would be an obligation to say that the human condition may not be considered in the wild, as we have argued in the book The General Intelligence of Human Nature.

Furthermore, it is the lifeblood of the human condition and the lifeblood of our body that are the lifeblood of the human condition, which is a watershed moment in the lifeblood of our body, which is a place where there is a world of awareness and care and compassion that exists in the world.

This is the first time that the human condition has been confirmed by human activity.

This has occurred as human activity in the body is one of the most important information technologies for human consciousness. Human consciousness is an important part of human consciousness, and humanity is a group of human beings. Human consciousness is a human form of consciousness
```

Brief assessment: the sample has locally fluent English and plausible article-like phrasing, but it repeats broad abstractions such as "human condition" and loses global specificity. This is consistent with the higher OWT validation loss and the relatively small 512MB training subset.

## OWT Training Result

- Run directory: `artifacts/runs/owt_512M_full32k/`
- Device: cuda
- Vocab size: 32000
- Batch size: 128
- Context length: 256
- Steps: 10000
- Processed tokens: 327,680,000
- Training tokens available: 122,868,632
- Validation tokens available: 66,401,098
- Number of parameters: 45,224,448
- Total training wall-clock time: 3623.46 sec
- Final train loss: 3.7241
- Final validation loss: 3.9342
- Best validation loss observed: 3.9342 at step 10000

## Learning Rate Divergence Run

The initial LR sweep covered `1e-4`, `3e-4`, `5e-4`, `1e-3`, and `3e-3`, all of which trained without numerical failure. To include an over-large LR case in the sweep, I added a short TinyStories run with `max_lr=1.0`, `min_lr=0.1`, `warmup_iters=10`, batch size 128, and the same model shape as the main TinyStories experiments.

Result:

- Run name: `lr_1e0_diverged`
- Steps: 120
- Step 20 train loss: 316.7358
- Step 20 validation loss: 338.6428
- Final train loss: 28.1719
- Final validation loss: 28.0294

This run is classified as diverged/unstable due to the immediate loss blow-up at high LR. The corresponding files are `logs/lr_sweep/lr_1e0_diverged.jsonl` and `logs/lr_sweep/lr_1e0_diverged_summary.json`.
