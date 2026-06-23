# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from typing import Protocol

import torch

from megatron.core.jit import jit_fuser
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version


class LayerNormInterface(Protocol):
    """Interface that all LayerNorm implementations should follow."""

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        """Forward method for a LayerNorm implementation."""
        ...


class LayerNormBuilder(Protocol):
    """A protocol showing how Modules are expected to construct LayerNorms."""

    def __call__(
        self, *, config: TransformerConfig, hidden_size: int, eps: float
    ) -> LayerNormInterface: ...


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ) -> LayerNormInterface:
        assert (
            not config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert not config.sequence_parallel, f"sequence parallel not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            assert is_torch_min_version(
                "2.4.0a0"
            ), 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        elif config.normalization == "L2Norm":
            norm_cls = torch.nn.L2Norm
        else:
            raise Exception("Only LayerNorm, RMSNorm and L2Norm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)


class L2Norm(torch.nn.Module, LayerNormInterface):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This module normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the actual L2 normalization.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the L2Norm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)


class SeeDNorm(torch.nn.Module, LayerNormInterface):
    """SeeDNorm: Self-Rescaled Dynamic Normalization (https://arxiv.org/abs/2510.22777).

    A drop-in replacement for RMSNorm whose per-channel gain is modulated by an input-dependent
    ("self-rescaled") term, so the layer can preserve input-norm information that a static RMSNorm
    gain discards::

        SeeDNorm(x) = [ tanh(seed(x)) * alpha + gamma ] * x / RMS(x)
        RMS(x)      = sqrt( mean(x**2, dim=-1) + eps )

    ``seed(x)`` is a per-token scalar (single head) or an ``n``-vector of per-head dot products
    (multi head). With ``num_heads = n`` the normalized dim ``D`` is split into ``n`` contiguous
    slices of size ``D / n``; head ``i`` contributes the scalar ``seed_i = <x_hi, beta_hi>`` and
    that scalar (after ``tanh``) broadcasts across head ``i``'s ``D / n`` channels of ``alpha``.
    Splitting reduces the dot-product variance per head (paper Thm. 3.2).

    Parameters (all shape ``[D]``, replicated across tensor-parallel ranks):
        * ``gamma`` -- static gain, init 1 (the usual RMSNorm weight).
        * ``alpha`` -- dynamic-gain scale, init 1.
        * ``beta``  -- seed projection, init 0. With ``beta = 0`` the seed is 0 and
          ``tanh(0) = 0``, so at initialization SeeDNorm is *exactly* ``gamma * x / RMS(x)`` --
          i.e. identical to RMSNorm -- which keeps it a safe drop-in.

    Args:
        config (TransformerConfig): provides ``seednorm_num_heads`` and ``sequence_parallel``.
        hidden_size (int): normalized feature dim ``D``.
        eps (float, optional): epsilon inside the RMS. Default: 1e-5.
    """

    def __init__(
        self, config: TransformerConfig, hidden_size: int, eps: float = 1e-5, **kwargs
    ):
        super().__init__()
        num_heads = getattr(config, "seednorm_num_heads", 1)
        assert hidden_size % num_heads == 0, (
            f"SeeDNorm: hidden_size ({hidden_size}) must be divisible by "
            f"seednorm_num_heads ({num_heads})."
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(hidden_size))  # static gain (RMSNorm weight)
        self.alpha = torch.nn.Parameter(torch.ones(hidden_size))  # dynamic-gain scale
        self.beta = torch.nn.Parameter(torch.zeros(hidden_size))  # seed projection (init 0)

        # Norm params are replicated across tensor-parallel ranks; mark them so their gradients are
        # all-reduced over the TP/sequence-parallel group, mirroring RMSNorm's `weight`.
        for param in (self.gamma, self.alpha, self.beta):
            setattr(param, 'sequence_parallel', config.sequence_parallel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SeeDNorm over the last dimension of ``x``."""
        in_dtype = x.dtype
        x = x.float()

        # RMS normalization over the full feature dim (fp32, like RMSNorm).
        x_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # Per-head seed: dot product of x with beta over each head's D/n slice, then tanh.
        beta = self.beta.float()
        if self.num_heads == 1:
            seed = (x * beta).sum(-1, keepdim=True)  # (..., 1)
            scale = torch.tanh(seed)  # (..., 1), broadcasts over D
        else:
            lead = x.shape[:-1]
            x_heads = x.reshape(*lead, self.num_heads, self.head_dim)  # (..., n, D/n)
            beta_heads = beta.reshape(self.num_heads, self.head_dim)  # (n, D/n)
            seed = (x_heads * beta_heads).sum(-1)  # (..., n)
            # Broadcast each head's scalar seed across its D/n channels -> (..., D).
            scale = torch.tanh(seed).repeat_interleave(self.head_dim, dim=-1)

        # Dynamic per-channel gain; at init (beta=0 -> scale=0) this is just gamma.
        gain = self.gamma.float() + self.alpha.float() * scale  # (..., D)
        return (gain * x_normed).to(in_dtype)
