import numpy as np
import torch
from torch import Tensor

def get_batch(token_ids, batch_size, context_length, device) -> tuple[Tensor, Tensor]:
    # return sampled input sequences and next-token targets (batch_size, context_length)
    # token_ids is a numpy array of shape (num_tokens,)
    # return (batch_size, context_length)
    max_start = token_ids.size - context_length
    indices = np.random.randint(0, max_start, size=batch_size)
    batch = [token_ids[i:(i + context_length)] for i in indices]
    next_tokens_batch = [token_ids[(i + 1):(i + context_length + 1)] for i in indices]

    return torch.Tensor(np.array(batch), device=device, dtype=token_ids.dtype), \
           torch.Tensor(np.array(next_tokens_batch), device=device, dtype=token_ids.dtype)

# TODO: check dtype
def get_batch_file(filename, batch_size, context_length, device):
    # Assume the tokens were saved with np.save and are of dtype=np.int64.
    token_ids = np.memmap(filename, mode='r', dtype=np.int64)

    return get_batch(token_ids, batch_size, context_length, device)
