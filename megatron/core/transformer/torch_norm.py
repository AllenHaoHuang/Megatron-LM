# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn

from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, input_scaling_init_value=0.5):
        super().__init__()
        # Not used but allows compatibility with LN and RMSNorm
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.input_scaling_init_value = input_scaling_init_value
        self.input_scaling = nn.Parameter(torch.ones(1) * input_scaling_init_value)

    def forward(self, x):
        x = torch.tanh(self.input_scaling * x)
        return x


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
    ):
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
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)
