from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import matplotlib.pyplot as plt

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, alpha=0.01, eps=1e-8, weight_decay=0.01):
        if lr < 0 or beta1 < 0 or beta2 < 0 or alpha < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(f"Invalid hyperparameters: lr: {lr}, beta1: {beta1}, beta2: {beta2}, alpha: {alpha}, eps: {eps}, weight_decay: {weight_decay}")

        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "alpha": alpha, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

        self.m = torch.zeros_like(params)
        self.v = torch.zeros_like(params)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.

                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.

                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss

def test_sgd(lr: float, iterations: int):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []
    for t in range(iterations):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        losses.append(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

    return losses

lr1_losses = test_sgd(1, 10)
lr10_losses = test_sgd(10, 10)
lr100_losses = test_sgd(100, 10)

plt.plot(lr1_losses, label="lr=1")
plt.plot(lr10_losses, label="lr=10")
plt.plot(lr100_losses, label="lr=100")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("SGD losses for different learning rates")
plt.legend()
plt.show()
