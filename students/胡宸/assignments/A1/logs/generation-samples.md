# Generation Samples

Generation used the trained checkpoint, matching tokenizer, temperature sampling and top-p filtering. Raw sample files live in `../assignment1-basics/runs/logs/`.

## TinyStories

Checkpoint: `runs/checkpoints/tinystories_gpu_baseline.pt`

```text
Once upon a time, there was a pretty bird named Bella. Bella lived in a big tree with her family. One day, Bella wanted to fly high in the sky. She asked her friend, a big bear named Ben, to help her.
"Ben, can you help me fly?" Bella asked. Ben was very kind and said, "Yes, I can help you!" So, they started to fly together. Bella flew high in the sky, and Ben watched. They were having so much fun.
After a while, Bella saw a big, soft nest on a tree. She said, "Ben, can you help me? The nest is very high up." Ben agreed, and they both flew to the nest. But when they got there, they saw a big, mean bird. The bird wanted to eat the nest, but Ben was too fast. He flew up to the nest and took the nest.
"Please, bird, do not take my nest!" Ben cried. The bird was sad and flew away. Bella and Ben were very sad. They did not get to fly the nest and had no friends. The moral of the story is: always be kind and help others, because they might not be able to catch you.
<|endoftext|>
```

The sample has recognizable TinyStories structure: simple characters, dialogue, conflict and a moral. Grammar is mostly fluent, but the plot is inconsistent near the end: Ben asks the bird not to take the nest after the text says Ben took it, and the moral is only loosely connected to the story.

## OpenWebText

Checkpoint: `runs/checkpoints/owt_gpu_full.pt`

```text
The history of artificial intelligence has been constantly linked to over the years in the wake of the US intelligence department.

In July, the NSA launched a new system that is taking place in Washington. The CIA has called the US to tell the US to do so in a way that, when there is a “progressive” legal system that would allow the Obama administration to pull it out of the way. The CIA said that the CIA had to be a “complete nation” under the war on the US and the US and Israel could not be on its territory. The Pentagon agreed that the Russians had the power to find out whether the CIA and the CIA could be able to support its intelligence agency, the CIA and the CIA, to avoid a national security adviser.

The CIA has now formally announced a second agreement.

The NSA have never set up a “modified” spy probe into the intelligence community, and claimed it was “a target of a terrorist attack” that was already targeted by Americans and the United States in 2005.
```

The OWT sample has news-like formatting and plausible local phrasing, but it repeats entities, makes unsupported factual claims and drifts semantically. This is consistent with the higher validation loss and the short 10000-step training budget.

## Factors Affecting Fluency

- Dataset/domain: TinyStories has simple and repetitive structure, so the model learns fluent local patterns more quickly than on broad OWT.
- Tokenizer: different vocabularies change sequence length, rare-word handling and per-token loss scale.
- Training budget: validation loss drops substantially with more processed tokens; the baseline still missed 1.45, while the high-LR continuation reached 1.3517.
- Architecture: RoPE and SwiGLU mattered in the ablation table; disabling RoPE had the largest loss increase.
- Sampling: higher temperature/top-p diversity can improve variety but worsens factuality and coherence when the model is undertrained.
