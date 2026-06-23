# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Differential Extension (DEX) for self-attention.

Implements the differential output adaptation from
"Understanding Differential Transformer Unchains Pretrained Self-Attentions"
(Kim et al., 2025; arXiv:2505.16333).

Instead of modifying the query-key circuit (as in DIFF Transformer, which requires a
second softmax), DEX reuses the existing softmax attention scores and applies a lightweight
learnable differential operation to the attention *output* ``O`` (per head ``h``):

    O   = softmax(Q Kᵀ / sqrt(d)) V                          # standard per-head output
    O'  = O - λ(t) · 𝟙(h ∈ ℋ) · f_D(O),   f_D(O) = O W_D     # implicit differential adaptation

where ``W_D ∈ ℝ^{d_v × d_v}`` is a single learnable projection shared across heads within a
layer (appendix F.3: ``O' = O (I - λ W_D)``), and ``ℋ`` is the per-layer subset of heads
selected for adaptation. The differential scalar follows the annealing schedule (Eq. 4):

    λ(t) = (1 - α) · (t / T) · λ_init + α · λ_learn,    α = min(1, t / T)

``λ_init`` is the depth-aware constant from the DIFF Transformer schedule
(``0.8 - 0.6·exp(-0.3·(layer_number - 1))``) and ``λ_learn`` is a learnable scalar
initialised near zero. At ``t = 0`` the schedule yields ``λ = 0`` so (combined with the
zero-initialised ``W_D``) the layer starts as an exact identity. NOTE: because ∇W_D ∝ λ and
∇λ_learn ∝ W_D, the point ``(W_D = 0, λ = 0)`` is a stationary point; the annealed ``λ_init``
term (which requires ``dex_anneal_steps > 0``) is what bootstraps the mechanism off zero.
"""

import math

import torch
from torch import Tensor
from torch.nn import Parameter, init

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

# Attribute flagging DEX parameters that are replicated across tensor-parallel ranks (like the
# per-head q/k layernorms). Their gradients are SUM-reduced across the TP group in
# megatron.core.distributed.finalize_model_grads._allreduce_non_tensor_model_parallel_grads.
DEX_REPLICATED_PARAM_ATTR = "dex_replicated"


def diff_lambda_init(layer_number: int) -> float:
    """Depth-aware ``λ_init`` from the DIFF Transformer schedule (``layer_number`` is 1-indexed)."""
    return 0.8 - 0.6 * math.exp(-0.3 * (layer_number - 1))


class DEX(MegatronModule):
    """Differential Extension applied to the concatenated per-head attention output.

    Shape-preserving: consumes and returns the attention output with shape
    ``[*, num_heads_per_partition * head_dim]`` (the layout produced by core attention, before
    the output projection), so it is agnostic to the attention backend and to GQA/MLA.

    Args:
        config: Transformer configuration.
        layer_number: 1-indexed layer number, used for the depth-aware ``λ_init``.
        head_dim: Per-head value dimension d_v that W_D acts on (``hidden_size_per_attention_head``
            for standard attention; ``v_head_dim`` for multi-latent attention).
        num_heads_per_partition: Number of (query) heads on this tensor-parallel rank.
        head_offset: Global index of this rank's first head (for head-selection mask slicing).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        head_dim: int,
        num_heads_per_partition: int,
        head_offset: int,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.head_dim = head_dim
        self.num_heads_per_partition = num_heads_per_partition
        self.head_offset = head_offset

        # Shared per-layer differential projection W_D ∈ R^{d_v × d_v}, replicated across TP.
        # `torch.empty` (not `torch.Tensor`) for correct device placement; see FusedLayerNorm.
        self.weight = Parameter(torch.empty(head_dim, head_dim))
        # Learnable differential scalar λ_learn (one per layer), replicated across TP.
        self.lambda_learn = Parameter(torch.empty(()))
        self.reset_parameters()

        # Flag both params as TP-replicated so their grads are SUM-reduced across the TP group.
        setattr(self.weight, DEX_REPLICATED_PARAM_ATTR, True)
        setattr(self.lambda_learn, DEX_REPLICATED_PARAM_ATTR, True)

        self.lambda_init = diff_lambda_init(layer_number)
        self.anneal_steps = int(config.dex_anneal_steps)

        # Current training step t. Persistent so the anneal can resume from a checkpoint; it is
        # also refreshed every training iteration from the training loop (see set_dex_iteration).
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=True)

        # Per-head selection mask 𝟙(h ∈ ℋ), sliced to this rank's tensor-parallel head partition.
        self.register_buffer("head_mask", self._build_head_mask(), persistent=False)

    def reset_parameters(self):
        """Zero-initialise W_D (so f_D(O) = 0 at start) and λ_learn (paper: init near zero)."""
        init.zeros_(self.weight)
        init.constant_(self.lambda_learn, float(self.config.dex_lambda_learn_init))

    def _build_head_mask(self) -> Tensor:
        """Build the per-head adaptation mask for this rank's local heads.

        ``'all'`` adapts every head. ``'half'`` adapts the second half of heads in each layer as
        a static, deterministic placeholder for the paper's data-driven (high-entropy /
        low-importance) selection, which needs an offline calibration pass over a pretrained model.
        """
        selection = self.config.dex_head_selection
        num_global_heads = self.config.num_attention_heads

        if selection == "all":
            global_mask = torch.ones(num_global_heads)
        elif selection == "half":
            k = num_global_heads // 2
            global_mask = torch.zeros(num_global_heads)
            global_mask[num_global_heads - k :] = 1.0
        else:
            raise ValueError(
                f"Invalid dex_head_selection: {selection!r} (expected 'all' or 'half')."
            )

        start = self.head_offset
        return global_mask[start : start + self.num_heads_per_partition].contiguous()

    def set_step(self, step: int):
        """Set the current training step ``t`` used by the λ annealing schedule."""
        self.step.fill_(int(step))

    def current_lambda(self, dtype: torch.dtype = torch.float32) -> Tensor:
        """Compute λ(t) per Eq. 4. Returns a 0-d tensor; grad flows through ``λ_learn``.

        Uses only tensor ops (no host sync on ``step``). With ``T <= 0`` annealing is disabled
        and λ = λ_learn (the zero-init learnable regime).
        """
        if self.anneal_steps <= 0:
            return self.lambda_learn.to(dtype)
        t = self.step.to(torch.float32)
        # frac = min(1, t / T) = α; the annealed term reuses the same (clamped) ratio.
        frac = torch.clamp(t / float(self.anneal_steps), max=1.0)
        lam = (1.0 - frac) * frac * self.lambda_init + frac * self.lambda_learn
        return lam.to(dtype)

    def forward(self, attn_output: Tensor) -> Tensor:
        """Apply DEX to the attention output ``[*, num_heads_per_partition * head_dim]``."""
        input_shape = attn_output.shape
        # [*, np * hd] -> [*, np, hd]
        heads = attn_output.view(*input_shape[:-1], self.num_heads_per_partition, self.head_dim)

        # f_D(O) = O W_D, applied per head with the shared W_D (cast to activation dtype, à la LN).
        diff = torch.matmul(heads, self.weight.to(heads.dtype))

        lam = self.current_lambda(heads.dtype)
        # Broadcast the per-head mask over leading dims and the head_dim: [*, np, 1].
        mask = self.head_mask.to(heads.dtype).view(
            *([1] * (heads.dim() - 2)), self.num_heads_per_partition, 1
        )

        heads = heads - lam * mask * diff
        return heads.reshape(*input_shape)


def set_dex_iteration(model, iteration: int):
    """Push the current training ``iteration`` into every DEX module in ``model``.

    Call once per training step (the iteration is checkpointed via the training state, so the
    λ schedule resumes correctly). ``model`` may be a single module or a list of model chunks,
    each possibly wrapped (DDP / Float16Module); ``modules()`` recurses through the wrappers.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for module in chunk.modules():
            if isinstance(module, DEX):
                module.set_step(iteration)
