# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for Differential Extension (DEX); see megatron.core.transformer.dex.

These exercise the DEX module in isolation and run on CPU, so they are usable in the local
``megatron-sn`` environment without GPU/TE. The full SelfAttention/MLA forward paths require a
GPU and are validated separately on the cluster/CI.
"""

import math
import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.transformer.dex import DEX, diff_lambda_init, set_dex_iteration
from megatron.core.transformer.transformer_config import TransformerConfig


@pytest.fixture(scope="module", autouse=True)
def _model_parallel():
    """Single-process (TP=1) model-parallel init on the gloo backend (CPU-friendly)."""
    import torch.distributed as dist

    created_pg = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12399")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
        created_pg = True
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    yield

    parallel_state.destroy_model_parallel()
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _config(**overrides):
    kwargs = dict(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,  # -> kv_channels = 4
        dex_enable=True,
        dex_anneal_steps=100,
        dex_head_selection="all",
        dex_lambda_learn_init=0.0,
        use_cpu_initialization=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _dex(config, layer_number, head_dim=None, num_heads_per_partition=None, head_offset=0):
    """Build a DEX module for the single-rank (TP=1) case, defaulting head_dim to kv_channels."""
    return DEX(
        config,
        layer_number=layer_number,
        head_dim=config.kv_channels if head_dim is None else head_dim,
        num_heads_per_partition=(
            config.num_attention_heads if num_heads_per_partition is None
            else num_heads_per_partition
        ),
        head_offset=head_offset,
    )


def test_diff_lambda_init_schedule():
    assert math.isclose(diff_lambda_init(1), 0.2, rel_tol=1e-6)  # 0.8 - 0.6 * exp(0)
    vals = [diff_lambda_init(l) for l in range(1, 8)]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # increasing with depth
    assert vals[-1] < 0.8


def test_init_is_identity_safe():
    """Zero-init W_D and zero-init lambda_learn => exact identity regardless of step."""
    dex = _dex(_config(dex_lambda_learn_init=0.0), layer_number=3)
    x = torch.randn(5, 2, 16)
    for step in (0, 50, 100, 250):
        dex.set_step(step)
        assert torch.allclose(dex(x), x, atol=0.0)


def test_lambda_at_step_zero_is_zero():
    dex = _dex(_config(dex_lambda_learn_init=0.7, dex_anneal_steps=100), layer_number=1)
    dex.set_step(0)
    assert torch.allclose(dex.current_lambda(), torch.zeros(()))


def test_lambda_schedule_values():
    # layer 1 -> lambda_init = 0.2; lambda_learn = 0.5; T = 100.
    dex = _dex(_config(dex_lambda_learn_init=0.5, dex_anneal_steps=100), layer_number=1)
    dex.set_step(50)  # frac = 0.5 -> (1-0.5)*0.5*0.2 + 0.5*0.5 = 0.30
    assert math.isclose(dex.current_lambda().item(), 0.30, rel_tol=1e-5)
    dex.set_step(100)  # frac = 1.0 -> lambda = lambda_learn
    assert math.isclose(dex.current_lambda().item(), 0.5, rel_tol=1e-5)
    dex.set_step(500)  # past T, clamped -> still lambda_learn
    assert math.isclose(dex.current_lambda().item(), 0.5, rel_tol=1e-5)


def test_anneal_disabled_uses_lambda_learn():
    dex = _dex(_config(dex_lambda_learn_init=0.7, dex_anneal_steps=0), layer_number=4)
    for step in (0, 10, 1000):
        dex.set_step(step)
        assert math.isclose(dex.current_lambda().item(), 0.7, rel_tol=1e-6)


def test_shape_preserved():
    dex = _dex(_config(), layer_number=1)
    dex.weight.data.normal_()
    dex.set_step(100)
    x = torch.randn(7, 3, 16)
    assert dex(x).shape == x.shape


def test_head_dim_decoupled_from_kv_channels():
    """MLA's value head dim (v_head_dim) generally differs from kv_channels; DEX must honor the
    explicitly passed head_dim rather than deriving it from kv_channels."""
    config = _config(hidden_size=32, num_attention_heads=8)  # kv_channels = 4
    dex = DEX(config, layer_number=1, head_dim=16, num_heads_per_partition=8, head_offset=0)
    assert dex.head_dim == 16
    assert dex.weight.shape == (16, 16)
    dex.weight.data.normal_()
    dex.set_step(100)
    x = torch.randn(3, 2, 8 * 16)  # attention output [s, b, n * v_head_dim]
    assert dex(x).shape == x.shape


def test_gqa_operates_on_query_heads():
    """Under GQA the output has one slot per query head; DEX never reads num_query_groups."""
    config = _config(hidden_size=32, num_attention_heads=8, num_query_groups=2)
    dex = _dex(config, layer_number=1)
    x = torch.randn(4, 2, 8 * config.kv_channels)
    assert dex(x).shape == x.shape


def test_head_mask_half_only_adapts_selected_heads():
    # 'half' selects the second half of heads: [0, 0, 1, 1] for 4 heads.
    dex = _dex(
        _config(dex_head_selection="half", dex_lambda_learn_init=1.0, dex_anneal_steps=0),
        layer_number=1,
    )
    dex.weight.data.fill_(1.0)  # deterministic, nonzero
    dex.set_step(0)  # anneal disabled -> lambda = 1.0

    x = torch.ones(3, 2, 16).view(3, 2, 4, 4)
    out = dex(x.view(3, 2, 16)).view(3, 2, 4, 4)

    assert torch.allclose(out[..., :2, :], x[..., :2, :])  # heads 0,1 unchanged
    # heads 2,3 selected -> O' = O - 1 * (O @ ones) = 1 - 4 = -3
    assert torch.allclose(out[..., 2:, :], torch.full_like(out[..., 2:, :], -3.0))


def test_gradients_flow_to_wd_and_lambda():
    dex = _dex(_config(dex_lambda_learn_init=0.5, dex_anneal_steps=100), layer_number=1)
    dex.weight.data.normal_()  # nonzero so lambda_learn receives gradient
    dex.set_step(100)  # lambda = lambda_learn = 0.5 (nonzero)
    x = torch.randn(4, 2, 16, requires_grad=True)
    dex(x).pow(2).sum().backward()
    assert dex.weight.grad is not None and dex.weight.grad.abs().sum() > 0
    assert dex.lambda_learn.grad is not None and dex.lambda_learn.grad.abs() > 0


def test_replicated_param_flags():
    """W_D and lambda_learn must be flagged for cross-TP grad all-reduce."""
    dex = _dex(_config(), layer_number=1)
    assert getattr(dex.weight, "dex_replicated", False) is True
    assert getattr(dex.lambda_learn, "dex_replicated", False) is True


def test_set_dex_iteration_walks_modules():
    container = torch.nn.ModuleList(
        [_dex(_config(), layer_number=1), _dex(_config(), layer_number=2)]
    )
    set_dex_iteration([container], 42)
    for m in container:
        assert int(m.step.item()) == 42
