import torch
from .base_layers import softmax

def top_p_sample(probs, top_p):
    sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    # remove tokens that exceed top_p, keep first token
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., :1] = False

    # zero out probabilities to remove and renormalize
    filtered = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
    return filtered / filtered.sum(dim=-1, keepdim=True)

def generate(model, prompt_tokens, max_tokens, context_length, eos_token_idx, device, temperature=0.0, top_p=0.9):
    model.eval()

    with torch.no_grad():
        tokens = torch.tensor(prompt_tokens, device=device)
        for _ in range(max_tokens):
            # truncate context length, run model, get last token logits
            tokens = tokens[-context_length:]
            logits = model(tokens)
            last_token_logits = logits[..., -1, :]

            # temperatured softmax
            probs = softmax(last_token_logits / temperature, dim=-1)

            # top_p sampling
            probs = top_p_sample(probs, top_p)
            next_token_index = torch.multinomial(probs, num_samples=1).item()
            next_token = torch.tensor([next_token_index], device=device)
            tokens = torch.cat([tokens, next_token])

            # check end of sequence
            if next_token == eos_token_idx:
                break

    return tokens
