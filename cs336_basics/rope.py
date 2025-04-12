import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_
from einops import einsum

class RotaryPositionalEmbedding:
   def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
      self.theta = theta
      self.d_k = d_k
      self.max_seq_len = max_seq_len
      self.device = device
   
   def get_angle(self, i, k, d):
      pass
   
   def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
      pass