from torch import nn, Tensor
from .attention import MultiheadSelfAttention
from .base_layers import Linear, RMSNorm, silu

class SwigluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()

        self.d_ff = d_ff if d_ff else int(round((8 / 3) * d_model / 64) * 64)

        self.w_1 = Linear(d_model, self.d_ff, device, dtype)
        self.w_2 = Linear(self.d_ff, d_model, device, dtype)
        self.w_3 = Linear(d_model, self.d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        silu_x = silu(self.w_1(x))
        inner_product = silu_x * self.w_3(x)
        return self.w_2(inner_product)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope_theta=None,
                 max_seq_len=None, token_positions=None, device=None, dtype=None):
        super().__init__()

        self.attention = MultiheadSelfAttention(d_model, num_heads, rope_theta=rope_theta,
                                                max_seq_len=max_seq_len, token_positions=token_positions,
                                                device=device, dtype=dtype)

        self.attention_norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.ffn = SwigluFFN(d_model, d_ff, device=device, dtype=dtype)
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        attention_output = x + self.attention(self.attention_norm(x))
        return attention_output + self.ffn(self.ffn_norm(attention_output))