import torch
from torch import Tensor
from einops import einsum
from .transformer import softmax

def scaled_dot_product_attention(d_k : int, Q: Tensor, K: Tensor, V: Tensor, mask=None):
   # Q = n x d_k, K = m x d_k, V = m x d_v
   # Mask = {True, False} (n x m)
   # transform (batch_size, ..., seq_len, d_k) -> (batch_size, ..., seq_len, d_v)
   pre_softmax = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / (d_k ** 0.5)
   if mask is not None:
      pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
   
   attention_weights = softmax(pre_softmax, dim=-1)
   return einsum(attention_weights, V, "... n m, ... m d_v -> ... n d_v")