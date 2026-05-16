# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


@jit_fuser
def compiled_xssslupr(x, alpha_p2, alpha_p3, alpha_n, beta):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    x = norm(x)
    return torch.where(x > 0,
                      alpha_p3 * x * x * x + alpha_p2 * x * x + beta * x,
                      alpha_n * x * torch.nn.functional.softsign(x) + beta * x)


class XSSSLUPR(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # Create vectors of length num_local_experts (or scalar if 1)
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), 0.8))
        self.alpha_p3 = nn.Parameter(torch.full((num_local_experts,), 0.4))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), 0.8 - 0.5))
        self.beta = nn.Parameter(torch.full((num_local_experts,), 0.5))

    def forward(self, x, tokens_per_expert=None):
        # Compute per‑expert parameters (positive)
        alpha_p2 = torch.abs(self.alpha_p2)          # (num_local_experts,)
        alpha_p3 = torch.abs(self.alpha_p3)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        beta = torch.abs(self.beta)

        if tokens_per_expert is None or self.num_local_experts == 1:
            # Broadcast scalar or (1,) to all tokens
            alpha_p2_t = alpha_p2
            alpha_p3_t = alpha_p3
            alpha_n_t = alpha_n
            beta_t = beta
        else:
            # Expand to per‑token values
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_p3_t = torch.repeat_interleave(alpha_p3, tpe_tensor).unsqueeze(-1)
            alpha_n_t = torch.repeat_interleave(alpha_n, tpe_tensor).unsqueeze(-1)
            beta_t = torch.repeat_interleave(beta, tpe_tensor).unsqueeze(-1)

        return compiled_xssslupr(x, alpha_p2_t, alpha_p3_t, alpha_n_t, beta_t)


@jit_fuser
def compiled_gxssslupr(x, alpha_p2, alpha_p3, alpha_n, beta):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    x = norm(x)
    return torch.where(x > 0,
                      alpha_p3 * x * x + alpha_p2 * x + beta,
                      alpha_n * torch.nn.functional.softsign(x) + beta)


class GXSSSLUPR(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), 0.8))
        self.alpha_p3 = nn.Parameter(torch.full((num_local_experts,), 0.4))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), 0.8 - 0.5))
        self.beta = nn.Parameter(torch.full((num_local_experts,), 0.5))

    def forward(self, x, tokens_per_expert=None):
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_p3 = torch.abs(self.alpha_p3)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        beta = torch.abs(self.beta)

        if tokens_per_expert is None or self.num_local_experts == 1:
            alpha_p2_t = alpha_p2
            alpha_p3_t = alpha_p3
            alpha_n_t = alpha_n
            beta_t = beta
        else:
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_p3_t = torch.repeat_interleave(alpha_p3, tpe_tensor).unsqueeze(-1)
            alpha_n_t = torch.repeat_interleave(alpha_n, tpe_tensor).unsqueeze(-1)
            beta_t = torch.repeat_interleave(beta, tpe_tensor).unsqueeze(-1)

        return compiled_gxssslupr(x, alpha_p2_t, alpha_p3_t, alpha_n_t, beta_t)


@jit_fuser
def compiled_polynorm(x, beta, alpha_p2, alpha_p3, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    norm_x = norm(x)
    norm_x2 = norm(torch.pow(x, 2))
    norm_x3 = norm(torch.pow(x, 3))
    return beta * norm_x + alpha_p2 * norm_x2 + alpha_p3 * norm_x3


class PolyNorm(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None, eps=1e-6):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # Create vectors of length num_local_experts
        self.beta = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.alpha_p3 = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.eps = eps

    def forward(self, x, tokens_per_expert=None):
        # Ensure parameters are positive (optional, as in reference)
        beta = torch.abs(self.beta)   # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_p3 = torch.abs(self.alpha_p3)

        if tokens_per_expert is None or self.num_local_experts == 1:
            # Broadcast scalar or (1,) to all tokens
            beta_t = beta
            alpha_p2_t = alpha_p2
            alpha_p3_t = alpha_p3
        else:
            # Expand to per‑token values
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            beta_t = torch.repeat_interleave(beta, tpe_tensor).unsqueeze(-1)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_p3_t = torch.repeat_interleave(alpha_p3, tpe_tensor).unsqueeze(-1)

        return compiled_polynorm(x, beta_t, alpha_p2_t, alpha_p3_t, self.eps)


@jit_fuser
def compiled_piecewise_polynorm(x, alpha_p2, alpha_p3, alpha_n2, beta, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    norm_x = norm(x)
    norm_x2 = norm(torch.pow(x, 2))
    norm_x3 = norm(torch.pow(x, 3))
    return beta * norm_x + alpha_p2 * norm_x2 * (x > 0) + alpha_n2 * norm_x2 * (x < 0) + alpha_p3 * norm_x3 * (x > 0)


class PiecewisePolyNorm(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None, eps=1e-6):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # Create vectors of length num_local_experts
        self.beta = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.alpha_p3 = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.alpha_n2 = nn.Parameter(torch.full((num_local_experts,), 0.33))
        self.eps = eps

    def forward(self, x, tokens_per_expert=None):
        # Ensure parameters are positive
        beta = torch.abs(self.beta)   # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_p3 = torch.abs(self.alpha_p3)
        alpha_n2 = torch.abs(self.alpha_n2)

        if tokens_per_expert is None or self.num_local_experts == 1:
            # Broadcast scalar or (1,) to all tokens
            beta_t = beta
            alpha_p2_t = alpha_p2
            alpha_p3_t = alpha_p3
            alpha_n2_t = alpha_n2
        else:
            # Expand to per‑token values
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            beta_t = torch.repeat_interleave(beta, tpe_tensor).unsqueeze(-1)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_p3_t = torch.repeat_interleave(alpha_p3, tpe_tensor).unsqueeze(-1)
            alpha_n2_t = torch.repeat_interleave(alpha_n2, tpe_tensor).unsqueeze(-1)

        return compiled_piecewise_polynorm(x, alpha_p2_t, alpha_p3_t, alpha_n2_t, beta_t, self.eps)


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Squared ReLU activation"""
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Quick GELU activation"""
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    """Fast GELU activation"""
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
