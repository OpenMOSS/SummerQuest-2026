import torch
from cs336_basics.model import softmax    

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_p=1.0,
             eos_token_id=None, context_length=256, device="cpu"):
    model.eval()
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0) 

    for _ in range(max_new_tokens):
        input_ids = ids[:, -context_length:]        
        logits = model(input_ids)                 
        logits = logits[0, -1, :]                   

        if temperature == 0:                        
            next_id = logits.argmax(dim=-1, keepdim=True) 
        else:
            probs = softmax(logits / temperature, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1)  

        ids = torch.cat([ids, next_id.unsqueeze(0)], dim=1)    

        if eos_token_id is not None and next_id.item() == eos_token_id:
            break

    return ids[0].tolist()

def top_p_filter(probs, top_p):

    if top_p >= 1.0:
        return probs                                  

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)  
    cumsum = torch.cumsum(sorted_probs, dim=-1)                    
    remove = cumsum - sorted_probs > top_p                        
    sorted_probs[remove] = 0.0                                    

    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs)               
    return filtered / filtered.sum()                            