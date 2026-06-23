# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Tests for SeeDNorm (Self-Rescaled Dynamic Normalization, https://arxiv.org/abs/2510.22777).

SeeDNorm is a pure-torch Megatron-Core norm, so these run on CPU (no GPU/TE/Triton needed). They
cover the module math (init reduces exactly to RMSNorm, multi-head seed locality, dtype/grad
behaviour), that `--seednorm` swaps SeeDNorm into ONLY the pre-attention / pre-MLP norm slots
(leaving QK and the base norm as RMSNorm), and the config guards.
"""
import types

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.torch_norm import SeeDNorm
from megatron.core.transformer.transformer_config import TransformerConfig

EPS = 1e-5
HIDDEN = 16


def _cfg(num_heads=1, sequence_parallel=False):
    """Minimal duck-typed config: SeeDNorm only reads these two attributes."""
    return types.SimpleNamespace(
        seednorm_num_heads=num_heads, sequence_parallel=sequence_parallel
    )


def _rmsnorm_ref(x, eps=EPS):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).type_as(x)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8, 16])
def test_init_equals_rmsnorm(num_heads):
    """beta=0 -> tanh(0)=0 -> gain=gamma=1, so SeeDNorm is exactly RMSNorm at init."""
    torch.manual_seed(0)
    x = torch.randn(4, 5, HIDDEN)
    out = SeeDNorm(_cfg(num_heads), HIDDEN, EPS)(x)
    torch.testing.assert_close(out, _rmsnorm_ref(x), rtol=0, atol=1e-6)


def test_dynamic_gain_changes_output_and_is_finite():
    torch.manual_seed(0)
    x = torch.randn(4, 5, HIDDEN)
    sn = SeeDNorm(_cfg(4), HIDDEN, EPS)
    with torch.no_grad():
        sn.beta.normal_(0, 0.5)
        sn.alpha.normal_(1.0, 0.1)
        sn.gamma.normal_(1.0, 0.1)
    out = sn(x)
    assert torch.isfinite(out).all()
    assert (out - _rmsnorm_ref(x)).abs().max() > 1e-3


def test_gradients_flow_to_all_params():
    torch.manual_seed(0)
    x = torch.randn(4, 5, HIDDEN)
    sn = SeeDNorm(_cfg(2), HIDDEN, EPS)
    with torch.no_grad():
        sn.beta.normal_(0, 0.5)  # ensure the alpha branch sees a nonzero seed
    sn(x).pow(2).sum().backward()
    for name in ("gamma", "alpha", "beta"):
        grad = getattr(sn, name).grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0, f"{name} received a zero gradient"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dtype_preserved(dtype):
    x = torch.randn(2, 3, HIDDEN, dtype=dtype)
    assert SeeDNorm(_cfg(2), HIDDEN, EPS)(x).dtype == dtype


def test_multi_head_seed_is_head_local():
    """A nonzero beta in head 0 only must perturb head 0's channels, not the others."""
    sn = SeeDNorm(_cfg(4), HIDDEN, EPS)
    head_dim = HIDDEN // 4
    with torch.no_grad():
        sn.beta.zero_()
        sn.beta[0] = 1.0
    x = torch.randn(1, 1, HIDDEN)
    diff = (sn(x) - _rmsnorm_ref(x)).abs().reshape(HIDDEN)
    assert diff[:head_dim].sum() > 1e-4
    assert diff[head_dim:].max() < 1e-6


def test_sequence_parallel_flag_propagates():
    for sp in (True, False):
        sn = SeeDNorm(_cfg(1, sequence_parallel=sp), HIDDEN, EPS)
        assert all(
            getattr(p, "sequence_parallel") is sp for p in (sn.gamma, sn.alpha, sn.beta)
        )


def test_non_divisible_num_heads_rejected():
    with pytest.raises(AssertionError):
        SeeDNorm(_cfg(5), HIDDEN, EPS)  # 16 % 5 != 0


# --- prenorm-only wiring --------------------------------------------------------------------

def test_seednorm_replaces_only_prenorm_slots():
    """--seednorm puts SeeDNorm in input_layernorm + pre_mlp_layernorm and NOWHERE else."""
    sub = get_gpt_layer_local_submodules(
        normalization="RMSNorm", seednorm=True, qk_layernorm=True
    )
    # The two residual-stream pre-norms become SeeDNorm...
    assert sub.input_layernorm is SeeDNorm
    assert sub.pre_mlp_layernorm is SeeDNorm
    # ...but the QK norm does NOT (it stays the RMSNorm builder even with qk_layernorm on).
    assert sub.self_attention.submodules.q_layernorm is not SeeDNorm
    assert sub.self_attention.submodules.k_layernorm is not SeeDNorm


def test_no_seednorm_keeps_rmsnorm_prenorm():
    sub = get_gpt_layer_local_submodules(normalization="RMSNorm", seednorm=False)
    assert sub.input_layernorm is not SeeDNorm
    assert sub.pre_mlp_layernorm is not SeeDNorm


def test_seednorm_leaves_sandwich_norm_alone():
    """With sandwich_norm on, the post-attn / post-MLP norms stay RMSNorm (not SeeDNorm)."""
    sub = get_gpt_layer_local_submodules(
        normalization="RMSNorm", seednorm=True, sandwich_norm=True
    )
    # Pre-norms are still swapped to SeeDNorm...
    assert sub.input_layernorm is SeeDNorm
    assert sub.pre_mlp_layernorm is SeeDNorm
    # ...but the sandwich (post) norms are active and are NOT SeeDNorm.
    assert sub.post_self_attn_layernorm is not SeeDNorm
    assert sub.post_mlp_layernorm is not SeeDNorm
    assert sub.post_self_attn_layernorm is not IdentityOp  # sandwich norm is actually present
    assert sub.post_mlp_layernorm is not IdentityOp


# --- config guards --------------------------------------------------------------------------

def _base_config(**overrides):
    base = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        normalization="RMSNorm",
        transformer_impl="local",
        seednorm=True,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def test_config_accepts_seednorm_local_rmsnorm():
    cfg = _base_config(seednorm_num_heads=2)
    assert cfg.seednorm and cfg.seednorm_num_heads == 2


def test_config_allows_qk_layernorm_with_seednorm():
    # QK norm is out of scope for SeeDNorm but must not be *blocked* -- it just stays RMSNorm.
    assert _base_config(qk_layernorm=True).seednorm is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"normalization": "LayerNorm"},  # SeeDNorm is an RMSNorm variant
        {"transformer_impl": "transformer_engine"},  # TE cannot represent SeeDNorm
        {"seednorm_num_heads": 5},  # must divide hidden_size (16)
        {"fused_residual_rmsnorm": True},  # residual fusion assumes plain RMSNorm
    ],
)
def test_config_rejects_unsupported_combinations(overrides):
    with pytest.raises(ValueError):
        _base_config(**overrides)
