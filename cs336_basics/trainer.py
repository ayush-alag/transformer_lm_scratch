import torch
from torch import Tensor


def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    # predictions: (batch_size, sequence_length, vocab_size)
    # targets: (batch_size, sequence_length)
    # log(sum(exp(o - max))) - (o - max)
    o_is = predictions[targets]
    o_is_max = o_is.max(dim=-1, keepdim=True)[0]
    stable_o_is = o_is - o_is_max
    exp_stable_o_is = torch.exp(stable_o_is)
    log_sum = torch.log(exp_stable_o_is.sum(dim=-1, keepdim=True))
    return log_sum - o_is_max[torch.arange(predictions.shape[0]), targets]
