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

    torch_dtype = np_dtype_to_torch_dtype(token_ids.dtype)
    return torch.tensor(np.array(batch), device=device, dtype=torch_dtype), \
        torch.tensor(np.array(next_tokens_batch), device=device, dtype=torch_dtype)

def np_dtype_to_torch_dtype(np_dtype):
    if np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.uint16:
        return int
    else:
        raise ValueError("Unsupported dtype: " + str(np_dtype))