import torch

def cross_entropy(logits, targets):

    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    losses = -log_probs[torch.arange(logits.shape[0]), targets]
    return losses.mean()