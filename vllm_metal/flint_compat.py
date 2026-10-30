# SPDX-License-Identifier: Apache-2.0
"""Accept Recursive Flint checkpoints on Transformers and vLLM releases.

Flint (``model_type=flint``, ``FlintForCausalLM``) landed in Transformers and
vLLM main after their latest releases. vllm-metal runs Flint through mlx-lm, so
both projects only need to accept the checkpoint config: this plugin registers a
Transformers config for ``flint`` and maps the architecture to vLLM's
Transformers MoE class, the way vLLM registers other MoE models it serves
through Transformers. The MLX path never builds that class.

TODO: remove once the pinned Transformers and vLLM releases know Flint.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

FLINT_MODEL_TYPE = "flint"
FLINT_ARCHITECTURE = "FlintForCausalLM"
_FLINT_VALIDATION_CLASS = (
    "vllm.model_executor.models.transformers:TransformersMoEForCausalLM"
)


def _flint_config_class() -> type[Any]:
    """Return a Transformers config for raw ``model_type=flint`` metadata."""
    from transformers.configuration_utils import PretrainedConfig

    class FlintCompatConfig(PretrainedConfig):
        """Decoder-only MoE with hybrid sliding/full attention.

        Sliding layers use RoPE; the full-attention layers listed in
        ``no_rope_layer_indices`` have no positional encoding. Every attention
        head has a learned sink logit. ``sliding_window`` counts the current
        token.
        """

        model_type = FLINT_MODEL_TYPE
        keys_to_ignore_at_inference = ["past_key_values"]

        def __init__(
            self,
            vocab_size: int = 150144,
            hidden_size: int = 2048,
            intermediate_size: int = 6144,
            num_hidden_layers: int = 49,
            num_attention_heads: int = 32,
            num_key_value_heads: int = 4,
            head_dim: int = 128,
            hidden_act: str = "silu",
            max_position_embeddings: int = 262144,
            initializer_range: float = 0.02,
            rms_norm_eps: float = 1e-6,
            use_cache: bool = True,
            tie_word_embeddings: bool = False,
            rope_theta: float = 10000.0,
            rope_parameters: dict[str, Any] | None = None,
            attention_bias: bool = False,
            attention_dropout: float = 0.0,
            use_sliding_window: bool = True,
            sliding_window: int | None = 513,
            layer_types: list[str] | None = None,
            no_rope_layer_indices: list[int] | None = None,
            moe_intermediate_size: int = 512,
            shared_expert_intermediate_size: int = 1024,
            num_experts_per_tok: int = 6,
            num_experts: int = 256,
            norm_topk_prob: bool = True,
            router_score_function: str = "sigmoid",
            mlp_only_layers: list[int] | None = None,
            **kwargs: Any,
        ) -> None:
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.head_dim = head_dim
            self.hidden_act = hidden_act
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.use_cache = use_cache
            self.rope_theta = rope_theta
            self.rope_parameters = rope_parameters or {
                "rope_type": "default",
                "rope_theta": rope_theta,
            }
            self.attention_bias = attention_bias
            self.attention_dropout = attention_dropout
            self.use_sliding_window = use_sliding_window
            self.sliding_window = sliding_window if use_sliding_window else None
            self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
            if len(self.layer_types) != num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries but "
                    f"num_hidden_layers is {num_hidden_layers}"
                )
            self.no_rope_layer_indices = no_rope_layer_indices or []
            self.moe_intermediate_size = moe_intermediate_size
            self.shared_expert_intermediate_size = shared_expert_intermediate_size
            self.num_experts_per_tok = num_experts_per_tok
            self.num_experts = num_experts
            self.norm_topk_prob = norm_topk_prob
            self.router_score_function = router_score_function
            self.mlp_only_layers = mlp_only_layers or []
            super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    return FlintCompatConfig


def _transformers_knows_flint(auto_config: Any) -> bool:
    try:
        auto_config.for_model(FLINT_MODEL_TYPE)
    except ValueError:
        return False
    return True


def _register_flint_config() -> None:
    from transformers import AutoConfig

    if _transformers_knows_flint(AutoConfig):
        return
    AutoConfig.register(FLINT_MODEL_TYPE, _flint_config_class(), exist_ok=True)
    logger.debug("Registered Flint config compatibility")


def _register_flint_architecture() -> None:
    from vllm.model_executor.models import ModelRegistry

    if FLINT_ARCHITECTURE in ModelRegistry.get_supported_archs():
        return
    ModelRegistry.register_model(FLINT_ARCHITECTURE, _FLINT_VALIDATION_CLASS)
    logger.debug("Registered Flint architecture compatibility")


def register() -> None:
    """Register Flint's missing config and architecture metadata (idempotent)."""
    _register_flint_config()
    _register_flint_architecture()
