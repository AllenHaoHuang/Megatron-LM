# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.jit import jit_fuser
from megatron.core.transformer.module import MegatronModule


# Trying to apply @jit_fuser / @torch.compile to XIELU class causes issues with sharded_state_dict naming
@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)


class XIELU(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.tensor(alpha_p_init).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.tensor(alpha_n_init - beta).unsqueeze(0))
        self.beta = nn.Parameter(torch.tensor(beta).unsqueeze(0))
        self.eps = torch.tensor(eps, device='cuda')

    def forward(self, x):
        alpha_p = torch.abs(self.alpha_p)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        return compiled_xielu(x, alpha_p, alpha_n, torch.abs(self.beta), self.eps)


@jit_fuser
def compiled_xssslur2(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * x * torch.nn.functional.softsign(x) + beta * x)


class XSSSLUR2(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.tensor(alpha_p_init).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.tensor(alpha_n_init - beta).unsqueeze(0))
        self.beta = nn.Parameter(torch.tensor(beta).unsqueeze(0))
        self.eps = torch.tensor(eps, device='cuda')

    def forward(self, x):
        alpha_p = torch.abs(self.alpha_p)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        return compiled_xssslur2(x, alpha_p, alpha_n, torch.abs(self.beta), self.eps)


@jit_fuser
def compiled_xssslupr(x, alpha_p1, alpha_p2, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p2 * x * x * x + alpha_p1 * x * x + beta * x,
                      alpha_n * x * torch.nn.functional.softsign(x) + beta * x)


class XSSSLUPR(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # Create vectors of length num_local_experts (or scalar if 1)
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_p_init))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), 0.4))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), alpha_n_init - beta))
        self.beta = nn.Parameter(torch.full((num_local_experts,), beta))
        self.eps = torch.tensor(eps, device='cuda')

    def forward(self, x, tokens_per_expert=None):
        # Compute per‑expert parameters (positive)
        alpha_p1 = torch.abs(self.alpha_p1)          # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        beta = torch.abs(self.beta)

        if tokens_per_expert is None or self.num_local_experts == 1:
            # Broadcast scalar or (1,) to all tokens
            alpha_p1_t = alpha_p1
            alpha_p2_t = alpha_p2
            alpha_n_t = alpha_n
            beta_t = beta
        else:
            # Expand to per‑token values
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            alpha_p1_t = torch.repeat_interleave(alpha_p1, tpe_tensor).unsqueeze(-1)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_n_t = torch.repeat_interleave(alpha_n, tpe_tensor).unsqueeze(-1)
            beta_t = torch.repeat_interleave(beta, tpe_tensor).unsqueeze(-1)

        return compiled_xssslupr(x, alpha_p1_t, alpha_p2_t, alpha_n_t, beta_t, self.eps)


@jit_fuser
def compiled_nxpr(x, alpha_p1, alpha_p2, alpha_n, beta=0.5, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm(torch.where(x > 0,
                      alpha_p2 * x * x * x + alpha_p1 * x * x + beta * x,
                      alpha_n * x * torch.nn.functional.softsign(x) + beta * x))


class NXPR(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p1 = nn.Parameter(torch.tensor(0.8).unsqueeze(0))
        self.alpha_p2 = nn.Parameter(torch.tensor(0.4).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.tensor(alpha_n_init - beta).unsqueeze(0))
        self.beta = nn.Parameter(torch.tensor(beta).unsqueeze(0))
        self.eps = torch.tensor(eps, device='cuda')
        self.act1_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        alpha_p1 = torch.abs(self.alpha_p1)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        return compiled_nxpr(x, alpha_p1, alpha_p2, alpha_n, torch.abs(self.beta), self.eps) * self.act1_scale


@jit_fuser
def compiled_gxssslur2(x, y, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * y + beta * y,
                      alpha_n * y * torch.nn.functional.softsign(x) + beta * y)


class GXSSSLUR2(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p = nn.Parameter(torch.tensor(alpha_p_init).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.tensor(alpha_n_init - beta).unsqueeze(0))
        self.beta = nn.Parameter(torch.tensor(beta).unsqueeze(0))
        self.eps = torch.tensor(eps, device='cuda')

    def forward(self, x, y):
        alpha_p = torch.abs(self.alpha_p)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        return compiled_gxssslur2(x, y, alpha_p, alpha_n, torch.abs(self.beta), self.eps)


class GXSSSLUPR(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_p_init))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), 0.4))
        self.alpha_n = nn.Parameter(torch.full((num_local_experts,), alpha_n_init - beta))
        self.beta = nn.Parameter(torch.full((num_local_experts,), beta))
        self.eps = torch.tensor(eps, device='cuda')

    def forward(self, x, y, tokens_per_expert=None):
        alpha_p1 = torch.abs(self.alpha_p1)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        beta = torch.abs(self.beta)

        if tokens_per_expert is None or self.num_local_experts == 1:
            alpha_p1_t = alpha_p1
            alpha_p2_t = alpha_p2
            alpha_n_t = alpha_n
            beta_t = beta
        else:
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            alpha_p1_t = torch.repeat_interleave(alpha_p1, tpe_tensor).unsqueeze(-1)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_n_t = torch.repeat_interleave(alpha_n, tpe_tensor).unsqueeze(-1)
            beta_t = torch.repeat_interleave(beta, tpe_tensor).unsqueeze(-1)

        return compiled_gxssslupr(x, y, alpha_p1_t, alpha_p2_t, alpha_n_t, beta_t, self.eps)


@jit_fuser
def compiled_ngxpr(x, y, alpha_p1, alpha_p2, alpha_n, beta=0.5, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return norm(torch.where(x > 0,
                      alpha_p2 * x * x * y + alpha_p1 * x * y + beta * y,
                      alpha_n * y * torch.nn.functional.softsign(x) + beta * y))


class NGXPR(MegatronModule):
    def __init__(self, config=None, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__(config=config)
        self.config = config
        self.alpha_p1 = nn.Parameter(torch.tensor(0.8).unsqueeze(0))
        self.alpha_p2 = nn.Parameter(torch.tensor(0.4).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.tensor(alpha_n_init - beta).unsqueeze(0))
        self.beta = nn.Parameter(torch.tensor(beta).unsqueeze(0))
        self.eps = torch.tensor(eps, device='cuda')
        self.act1_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, y):
        alpha_p1 = torch.abs(self.alpha_p1)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_n = torch.abs(self.beta) + torch.abs(self.alpha_n)
        return compiled_ngxpr(x, y, alpha_p1, alpha_p2, alpha_n, torch.abs(self.beta), self.eps) * self.act1_scale


@jit_fuser
def compiled_polyrelu(x, alpha_p1, alpha_p2, alpha_p3):
    relu_x = F.relu(x)
    return alpha_p1 * relu_x + alpha_p2 * torch.pow(relu_x, 2) + alpha_p3 * torch.pow(relu_x, 3)


class PolyReLU(MegatronModule):
    def __init__(self, config=None, alpha_init=0.33):
        super().__init__(config=config)
        self.alpha_p1 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p2 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p3 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))

    def forward(self, x):
        return compiled_polyrelu(x, self.alpha_p1, self.alpha_p2, self.alpha_p3)


@jit_fuser
def compiled_polynorm(x, alpha_p1, alpha_p2, alpha_p3, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return alpha_p1 * norm(x) + alpha_p2 * norm(x * x) + alpha_p3 * norm(x * x * x)


class PolyNorm(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None, alpha_init=0.33, eps=1e-6):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        # Create vectors of length num_local_experts
        self.alpha_p1 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_p2 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.alpha_p3 = nn.Parameter(torch.full((num_local_experts,), alpha_init))
        self.eps = eps

    def forward(self, x, tokens_per_expert=None):
        # Ensure parameters are positive (optional, as in reference)
        alpha_p1 = torch.abs(self.alpha_p1)   # (num_local_experts,)
        alpha_p2 = torch.abs(self.alpha_p2)
        alpha_p3 = torch.abs(self.alpha_p3)

        if tokens_per_expert is None or self.num_local_experts == 1:
            # Broadcast scalar or (1,) to all tokens
            alpha_p1_t = alpha_p1
            alpha_p2_t = alpha_p2
            alpha_p3_t = alpha_p3
        else:
            # Expand to per‑token values
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            tpe_tensor = torch.tensor(tokens_per_expert, device=x.device)
            alpha_p1_t = torch.repeat_interleave(alpha_p1, tpe_tensor).unsqueeze(-1)
            alpha_p2_t = torch.repeat_interleave(alpha_p2, tpe_tensor).unsqueeze(-1)
            alpha_p3_t = torch.repeat_interleave(alpha_p3, tpe_tensor).unsqueeze(-1)

        return compiled_polynorm(x, alpha_p1_t, alpha_p2_t, alpha_p3_t, self.eps)


@jit_fuser
def compiled_polynorm1(x, y, alpha_p1, alpha_p2, alpha_p3, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return alpha_p1 * norm(y) + alpha_p2 * norm(x * y) + alpha_p3 * norm(x * x * y)


class PolyNorm1(MegatronModule):
    def __init__(self, config=None, alpha_init=0.33, eps=1e-6):
        super().__init__(config=config)
        self.alpha_p1 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p2 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p3 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.eps = eps

    def forward(self, x, y):
        return compiled_polynorm1(x, y, self.alpha_p1, self.alpha_p2, self.alpha_p3, self.eps)


@jit_fuser
def compiled_polynorm2(x, y, alpha_p1, alpha_p2, alpha_p3, alpha_p4, alpha_p5, alpha_p6, alpha_p7, alpha_p8, alpha_p9, eps=1e-6):
    def norm(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return alpha_p1 * norm(x) + alpha_p2 * norm(y) + alpha_p3 * norm(x * y) + alpha_p4 * norm(x * x) + alpha_p5 * norm(y * y) + alpha_p6 * norm(x * x * x) + alpha_p7 * norm(x * x * y) + alpha_p8 * norm(x * y * y) + alpha_p9 * norm(y * y * y)
    
class PolyNorm2(MegatronModule):
    def __init__(self, config=None, alpha_init=0.1111, eps=1e-6):
        super().__init__(config=config)
        self.alpha_p1 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p2 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p3 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p4 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p5 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p6 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p7 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p8 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.alpha_p9 = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))
        self.eps = eps

    def forward(self, x, y):
        return compiled_polynorm2(x, y, self.alpha_p1, self.alpha_p2, self.alpha_p3, self.alpha_p4, self.alpha_p5, self.alpha_p6, self.alpha_p7, self.alpha_p8, self.alpha_p9, self.eps)


@jit_fuser
def sss(x):
    return 0.5 * (torch.nn.functional.softsign(x) + 1)


class SSS(MegatronModule):
    def __init__(self, config=None):
        super().__init__(config=config)

    def forward(self, x):
        return sss(x)


@jit_fuser
def ssslu(x):
    return (0.5 * (torch.nn.functional.softsign(x) + 1)) * x


class SSSLU(MegatronModule):
    def __init__(self, config=None):
        super().__init__(config=config)

    def forward(self, x):
        return ssslu(x)


@jit_fuser
def sssglu(x, y):
    return (0.5 * (torch.nn.functional.softsign(x) + 1)) * x * y


class SSSGLU(MegatronModule):
    def __init__(self, config=None):
        super().__init__(config=config)

    def forward(self, x, y):
        return sssglu(x, y)


@jit_fuser
def xsss(x, alpha):
    return alpha * torch.nn.functional.softsign(x) + 0.5


class XSSS(MegatronModule):
    def __init__(self, config=None, alpha_init=0.8):
        super().__init__(config=config)
        self.config = config
        self.alpha = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))

    def forward(self, x):
        return xsss(x, self.alpha)


@jit_fuser
def xssslu(x, alpha):
    return (alpha * torch.nn.functional.softsign(x) + 0.5) * x


class XSSSLU(MegatronModule):
    def __init__(self, config=None, alpha_init=0.8):
        super().__init__(config=config)
        self.config = config
        self.alpha = nn.Parameter(torch.tensor(alpha_init).unsqueeze(0))

    def forward(self, x):
        return xssslu(x, self.alpha)


@jit_fuser
def xsssglu(x, y, alpha_per_token):
    return (alpha_per_token * torch.nn.functional.softsign(x) + 0.5) * x * y

class XSSSGLU(MegatronModule):
    def __init__(self, num_local_experts: int = 1, config=None, alpha_init=0.8):
        super().__init__(config=config)
        self.num_local_experts = num_local_experts
        self.alpha_offset = nn.Parameter(torch.full((num_local_experts,), alpha_init - 0.5))

    def forward(self, x, y, tokens_per_expert=None):
        alpha_experts = 0.5 + torch.abs(self.alpha_offset)  # shape (num_local_experts,)
        if tokens_per_expert is None or self.num_local_experts == 1:
            alpha_per_token = alpha_experts  # scalar or (1,) broadcasts
        else:
            if isinstance(tokens_per_expert, torch.Tensor):
                tokens_per_expert = tokens_per_expert.tolist()
            alpha_per_token = torch.repeat_interleave(
                alpha_experts, torch.tensor(tokens_per_expert, device=x.device)
            ).unsqueeze(-1)
        return xsssglu(x, y, alpha_per_token)


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
