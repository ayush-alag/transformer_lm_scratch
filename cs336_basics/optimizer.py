from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
# import matplotlib.pyplot as plt

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0 or betas[0] < 0 or betas[1] < 0 or eps < 0 or weight_decay < 0:
            raise ValueError(f"Invalid hyperparameters: lr: {lr}, betas: {betas}, eps: {eps}, weight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                grad = p.grad.data

                # update m and v
                state["m"] = beta1 * m + (1 - beta1) * grad
                state["v"] = beta2 * v + (1 - beta2) * grad * grad

                ratio = ((1 - (beta2 ** t)) ** 0.5) / (1 - (beta1 ** t))
                lr_t = lr * ratio

                p.data -= lr_t * state["m"] / (state["v"] ** 0.5 + eps)
                p.data *= (1 - lr * weight_decay)
                state["t"] = t + 1

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
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

def test_sgd(lr: float, iterations: int):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    losses = []
    for t in range(iterations):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.cpu().item())
        loss.backward()
        opt.step()

    return losses

def learning_rate_schedule(t, a_max, a_min, t_w, t_c):
    if t < t_w:
        return a_max * (t / t_w)
    elif t_w <= t <= t_c:
        return a_min + 0.5 * (a_max - a_min) * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w)))
    else:
        return a_min

def grad_clipping(parameters, max_l2_norm, eps=1e-6):
    params_with_grad = [p for p in parameters if p.grad is not None]

    if not params_with_grad:
        return

    total_norm_sq = sum(p.grad.data.pow(2).sum() for p in params_with_grad)
    global_norm = total_norm_sq.sqrt().item()

    if global_norm > max_l2_norm:
        for p in params_with_grad:
            p.grad.data *= max_l2_norm / (global_norm + eps)

# lr1_losses = test_sgd(1, 10)
# lr10_losses = test_sgd(10, 10)
# lr100_losses = test_sgd(100, 10)

# plt.plot(lr1_losses, label="lr=1")
# plt.plot(lr10_losses, label="lr=10")
# plt.plot(lr100_losses, label="lr=100")
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("SGD losses for different learning rates")
# plt.legend()
# plt.show()
