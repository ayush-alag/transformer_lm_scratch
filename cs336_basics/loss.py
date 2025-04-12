import torch
from torch import Tensor


def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    # predictions: (..., vocab_size)
    # targets: (..., )
    # log(sum(exp(o - max))) - (o - max)
    stable_predictions = predictions - predictions.max(dim=-1, keepdim=True)[0]
    exp_stable_predictions = torch.exp(stable_predictions)
    log_sum = torch.log(exp_stable_predictions.sum(dim=-1, keepdim=True))

    negative_log_likelihood = log_sum - stable_predictions

    target_log_likelihood = negative_log_likelihood.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return target_log_likelihood.mean()

def perplexity(predictions: Tensor, targets: Tensor) -> Tensor:
    return torch.exp(cross_entropy_loss(predictions, targets))