import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn


class VarianceSchedule(nn.Module):
    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        self.num_steps = num_steps
        T = num_steps
        t = torch.arange(0, num_steps + 1, dtype=torch.float)
        f_t = torch.cos((np.pi / 2) * ((t / T) + s) / (1 + s)) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("sigmas", sigmas)

    if TYPE_CHECKING:
        betas: torch.Tensor
        alphas: torch.Tensor
        sigmas: torch.Tensor
        alpha_bars: torch.Tensor


class EuclideanDiffusion(nn.Module):
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x_0:  (*batch, dim)
            t:  (*batch)
        """
        alpha_bar: torch.Tensor = self.alpha_bars[t]

        c0 = torch.sqrt(alpha_bar)[..., None]
        c1 = torch.sqrt(1 - alpha_bar)[..., None]

        noise = torch.randn_like(x_0)
        x_noisy = c0 * x_0 + c1 * noise
        return x_noisy, noise

    def denoise(self, x_t: torch.Tensor, eps: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x_t:  (*batch, dim)
            eps:  (*batch, dim)
            t:  (*batch)
        """
        t = t[..., None]
        alpha = self.alphas[t].clamp_min(self.alphas[-2])
        alpha_bar = self.alpha_bars[t]
        sigma = self.sigmas[t]

        c0 = 1.0 / torch.sqrt(alpha + 1e-8)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8)

        z = torch.where(
            (t > 1).expand_as(x_t),
            torch.randn_like(x_t),
            torch.zeros_like(x_t),
        )

        p_next = c0 * (x_t - c1 * eps) + sigma * z
        return p_next


class BernoulliDiffusion(VarianceSchedule):
    @dataclasses.dataclass
    class _Sample:
        prob: torch.Tensor
        value: torch.Tensor

        def __iter__(self):
            return iter((self.prob, self.value))

        def __getitem__(self, item):
            return (self.prob, self.value)[item]

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x_0:  Ground truth binary values (0/1), LongTensor, (*batch)
            t:  (*batch)
        Returns:
            p_t:    Probability, (*batch)
            x_t:    Noisy sample, LongTensor, (*batch)
        """
        p_0 = x_0.float()
        alpha_bar = self.alpha_bars[t]
        p_t = (alpha_bar * p_0) + ((1 - alpha_bar) * 0.5)  # 0.5: uniform noise
        x_t = torch.bernoulli(p_t)
        return self._Sample(prob=p_t, value=x_t)

    def posterior(self, p_t: torch.Tensor, p_0: torch.Tensor, t: torch.Tensor):
        """
        Args:
            p_t:  (*batch).
            p_0:  (*batch).
            t:  (*batch).
        Returns:
            theta:  Posterior probability at (t-1)-th step, (*batch).
        """
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bars[t]

        theta_a = ((alpha * p_t) + (1 - alpha) * 0.5) * ((alpha_bar * p_0) + (1 - alpha_bar) * 0.5)
        theta_b = ((alpha * (1 - p_t)) + (1 - alpha) * 0.5) * ((alpha_bar * (1 - p_0)) + (1 - alpha_bar) * 0.5)
        theta = theta_a / (theta_a + theta_b)
        return theta

    def denoise(self, x_t: torch.Tensor, p_0_pred: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x_t:  Binary LongTensor (*batch).
            c_0_pred:  Normalized probability predicted by networks, ranges [0, 1], (*batch).
            t:  (*batch).
        Returns:
            post:  Posterior probability at (t-1)-th step, (*batch).
            x_next:  Sample at (t-1)-th step, LongTensor, (*batch).
        """
        p_t = x_t.float()
        post = self.posterior(p_t=p_t, p_0=p_0_pred, t=t)  # (N, L, K)
        x_next = torch.bernoulli(post)
        return self._Sample(prob=post, value=x_next)


class NoiseConditionedMLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int | None = None):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        dim_hidden = dim_hidden or dim_out
        self.dim_hidden = dim_hidden
        self.mlp = nn.Sequential(
            nn.Linear(dim_in + 3, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, var_sched: VarianceSchedule):
        """
        Args:
            x:  (*batch, dim_in)
            t:  (*batch)
        """
        t_rel = (t.to(dtype=x.dtype)[..., None] / var_sched.num_steps) * 2 * np.pi  # (*batch, 1)
        y = self.mlp(torch.cat([x, t_rel, torch.sin(t_rel), torch.cos(t_rel)], dim=-1))
        return y
