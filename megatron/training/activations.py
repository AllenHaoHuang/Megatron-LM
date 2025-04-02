# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


# Trying to apply @jit_fuser / @torch.compile to XIELU class causes issues with sharded_state_dict naming
@jit_fuser
def compiled_xieplu(x, alpha_n, alpha_p2, alpha_p3, alpha_p4, alpha_p5, alpha_p6, beta=0.5, eps=-1e-6):
    x_2 = x * x          # x^2
    x_3 = x_2 * x        # x^3 
    x_4 = x_3 * x        # x^4
    x_5 = x_4 * x        # x^5
    x_6 = x_5 * x        # x^6
    return torch.where(x > 0,
                      alpha_p6 * x_6 + alpha_p5 * x_5 + alpha_p4 * x_4 + alpha_p3 * x_3 + alpha_p2 * x_2 + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)


@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)


@jit_fuser
def compiled_xiprelu(x, alpha_p, alpha_n, beta=0.5):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * x * x + beta * x)


@jit_fuser
def compiled_xiprelup(x, alpha_p, alpha_n, power, beta=0.5, eps=1e-6):
    x_power = torch.pow(torch.max(torch.abs(x), eps), power)
    return torch.where(x > 0,
                      alpha_p * x_power + beta * x,
                      alpha_n * x_power + beta * x)


class XIEPLU(MegatronModule):
    def __init__(self, config=None, alpha_p2_init=1/2, alpha_p3_init=1/6, alpha_p4_init=1/24, alpha_p5_init=1/120, alpha_p6_init=1/720, alpha_n_init=0.8, beta=0.5):
        super().__init__(config)
        self.config = config
        self.alpha_p2 = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p2_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_p3 = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p3_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_p4 = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p4_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_p5 = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p5_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_p6 = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p6_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.beta = beta

    def forward(self, x):
        alpha_n = F.softplus(self.alpha_n)
        alpha_p2 = F.softplus(self.alpha_p2)
        alpha_p3 = F.softplus(self.alpha_p3)
        alpha_p4 = F.softplus(self.alpha_p4)
        alpha_p5 = F.softplus(self.alpha_p5)
        alpha_p6 = F.softplus(self.alpha_p6)
        return compiled_xieplu(x, alpha_n, alpha_p2, alpha_p3, alpha_p4, alpha_p5, alpha_p6, self.beta)


class XIELU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device='cuda')

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)


class XIPReLU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5):
        super().__init__(config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.beta = beta

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = F.softplus(self.alpha_n)
        return compiled_xiprelu(x, alpha_p, alpha_n, self.beta)


class XIPReLUP(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, power_init=2, beta=0.5, eps=1e-6):
        super().__init__(config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.power = nn.Parameter(torch.log(torch.exp(torch.tensor(power_init - 1.0, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device='cuda')

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = F.softplus(self.alpha_n)
        power = 1 + F.softplus(self.power)
        return compiled_xiprelup(x, alpha_p, alpha_n, power, self.beta, self.eps)


@jit_fuser
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(F.relu(x), 2)


@jit_fuser
def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


@jit_fuser
def fast_gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
