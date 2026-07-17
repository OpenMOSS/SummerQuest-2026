import argparse, pickle
import torch
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW, load_checkpoint
from cs336_basics.decode import generate

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--merges", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = TransformerLM(args.vocab_size, args.context_length, args.d_model,
                          args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(args.device)
    dummy_opt = AdamW(model.parameters(), lr=1e-3)
    load_checkpoint(args.checkpoint, model, dummy_opt)

    with open(args.vocab, "rb") as f:
        vocab = pickle.load(f)
    with open(args.merges, "rb") as f:
        merges = pickle.load(f)
    tok = Tokenizer(vocab, merges, ["<|endoftext|>"])


    prompt_ids = tok.encode(args.prompt)

    eos_id = tok.encode("<|endoftext|>")[0]

    out_ids = generate(model, prompt_ids, args.max_tokens,
                       temperature=args.temperature, top_p=args.top_p,
                       eos_token_id=eos_id, context_length=args.context_length,
                       device=args.device)
    print("=" * 60)
    print(tok.decode(out_ids))
    print("=" * 60)

if __name__ == "__main__":
    main()