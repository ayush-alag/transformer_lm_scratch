import torch
from torch import Tensor, nn
from einops import einsum, rearrange
from .base_layers import softmax, Linear
from .rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(d_k: int, Q: Tensor, K: Tensor, V: Tensor, mask=None):
    # Q = n x d_k, K = m x d_k, V = m x d_v
    # Mask = {True, False} (n x m)
    # transform (batch_size, ..., seq_len, d_k) -> (batch_size, ..., seq_len, d_v)
    pre_softmax = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / (d_k**0.5)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(pre_softmax, dim=-1)
    return einsum(attention_weights, V, "... n m, ... m d_v -> ... n d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        rope_theta=None,
        max_seq_len=None,
        token_positions=None,
        device=None,
        dtype=None,
        no_rope=False,
    ):
        super().__init__()

        self.d_kv = d_model // num_heads
        self.num_heads = num_heads
        self.device = device

        self.w_qkv = Linear(
            d_model, 3 * num_heads * self.d_kv, device=device, dtype=dtype
        )  # store q, k, v together
        self.w_o = Linear(self.d_kv * num_heads, d_model, device=device, dtype=dtype)

        if rope_theta and max_seq_len and not no_rope:
            self.rope = RotaryPositionalEmbedding(
                theta=rope_theta,
                d_k=self.d_kv,
                max_seq_len=max_seq_len,
                device=device,
            )

            assert token_positions is not None
        else:
            self.rope = None

        # only to be used with rope
        self.token_positions = token_positions

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        # upper triangular portion (excluding the diagonal) is False
        # and the lower triangular portion (including the diagonal) is True
        mask = torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool)
        return torch.tril(mask, diagonal=0)

    # b = 4, seq = 12, d_model = 64, heads = 4
    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x_qkv = self.w_qkv(x)  # w_qx, w_kx, w_vx
        x_qkv = rearrange(
            x_qkv,
            "b s (three heads kv) -> three b heads s kv",
            three=3,
            heads=self.num_heads,
        )
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]

        if self.rope:
            q = self.rope.forward(q, self.token_positions)
            k = self.rope.forward(k, self.token_positions)

        mask = self.create_causal_mask(x.shape[1])
        attention = scaled_dot_product_attention(self.d_kv, q, k, v, mask)
        attention = rearrange(attention, "b h s d -> b s (h d)")

        return self.w_o(attention)
