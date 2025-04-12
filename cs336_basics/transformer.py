import torch
from torch import nn, Tensor
from torch.nn.init import trunc_normal_
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # Construct a linear transformation module. This function should accept the following parameters:
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # call the superclass constructor
        super().__init__()

        # initialize the weight, no bias term
        stdev = (2 / (in_features + out_features)) ** 0.5
        max_normal = 3 * stdev
        min_normal = -max_normal

        zero_weights_transposed = torch.zeros([in_features, out_features], device=device, dtype=dtype)
        self.weights_transposed = nn.Parameter(trunc_normal_(zero_weights_transposed, 0, stdev, min_normal, max_normal))

    def forward(self, x: Tensor) -> Tensor:
        # Apply the linear transformation to theinput.
        return einsum(x, self.weights_transposed, "... d_in, d_in d_out -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        # num_embeddings: size of the vocabulary
        # embedding_dim: the size of the embedding dimension (d_model)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        # call the superclass constructor
        super().__init__()

        # initialize the weight, no bias term
        zero_embedding_mat = torch.zeros([self.num_embeddings, self.embedding_dim], device=device, dtype=dtype)
        self.embedding_mat = nn.Parameter(trunc_normal_(zero_embedding_mat, 0, 1, -3, 3))

    def forward(self, token_ids: Tensor) -> Tensor:
        # look up the embeddings for the token ids
        return self.embedding_mat[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        self.eps = eps
        self.device = device
        self.dtype = dtype

        super().__init__()

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = self.gain * (x / rms)
        return result.to(in_dtype)

class SwigluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        
        # TODO: how to use d_ff??
        self.d_ff = d_ff if d_ff else int(round((8 / 3) * d_model / 64) * 64)
        
        self.w_1 = Linear(d_model, self.d_ff, device, dtype)
        self.w_2 = Linear(self.d_ff, d_model, device, dtype)
        self.w_3 = Linear(d_model, self.d_ff, device, dtype)
    
    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, sequence_length, d_model) -> (batch_size, sequence_length, d_model)
        silu_x = silu(self.w_1.forward(x))
        inner_product = silu_x * self.w_3.forward(x)
        return self.w_2.forward(inner_product)
    
def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def softmax(x: Tensor, dim: int) -> Tensor:
    # get the max for each slice (other dimensions) for dimension i
    x_max = x.max(dim=dim, keepdim=True)[0]
    stable_x = x - x_max
    exp_stable_x = torch.exp(stable_x)
    
    return exp_stable_x / exp_stable_x.sum(dim=dim, keepdim=True)