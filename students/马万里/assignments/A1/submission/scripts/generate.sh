#!/bin/bash
export PYTHONPATH=/remote-home1/wlma/work/assignment1-basics:$PYTHONPATH
python scripts/generate.py \
    --vocab_path data/tinystories_tokenizer/vocab.pkl \
    --merges_path data/tinystories_tokenizer/merges.pkl \
    --checkpoint_path data/checkpoints/tinystories_baseline/final.pt \
    --prompt "Once upon a time" \
    --max_tokens 256 \
    --temperature 1.0 \
    --top_p 0.9 \
    --device cuda:0