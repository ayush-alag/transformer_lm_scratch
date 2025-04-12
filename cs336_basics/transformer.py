from torch import nn, Tensor
from .attention import MultiheadSelfAttention
from .base_layers import Linear, RMSNorm, silu, Embedding, softmax

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

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers,
                 d_model, num_heads, d_ff, rope_theta=None,
                 token_positions=None, device=None, dtype=None):
        super().__init__()

        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, rope_theta=rope_theta,
                                           max_seq_len=context_length, token_positions=token_positions,
                                           device=device, dtype=dtype))

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.embedding(x)
        output = embedding
        for i in range(len(self.transformer_blocks)):
            output = self.transformer_blocks[i](output)

        output = self.ln_final(output)
        return self.lm_head(output)
