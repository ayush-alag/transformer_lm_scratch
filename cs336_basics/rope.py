import torch
from torch import nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # call the superclass constructor
        super().__init__()

        self.populate_trig()

    def populate_trig(self):
        pos = torch.arange(self.max_seq_len, device=self.device).unsqueeze(1)
        ks = torch.arange(self.d_k // 2, device=self.device, dtype=torch.float32)
        inv_freq = self.theta ** (-(2 * ks / self.d_k))

        angles = pos * inv_freq
        self.cosines = angles.cos()
        self.sines = angles.sin()

        self.register_buffer("cos_vals", self.cosines, persistent=False)
        self.register_buffer("sin_vals", self.sines, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # for each token position we get the corresponding list of sines/cosines (size k)
        token_positions = torch.arange(x.shape[-2])
        cos = self.cos_vals[token_positions]
        sin = self.sin_vals[token_positions]
        x = rearrange(
            x,
            "... seq_len (d_half two) -> ... seq_len d_half two",
            two=2,
            d_half=self.d_k // 2,
        )
        x1 = x[..., 0] * cos - x[..., 1] * sin  # rotated first coordinate
        x2 = x[..., 0] * sin + x[..., 1] * cos  # rotated second coordinate

        return rearrange([x1, x2], "two ... seq_len d_half -> ... seq_len (d_half two)")
