from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn import ModuleList
from transformers import AutoProcessor

try:
    from transformers import Idefics3ForConditionalGeneration
except ImportError as exc:  # pragma: no cover - surfaced during import time
    raise ImportError("SmolVLM requires transformers with idefics3 support.") from exc

from vlm_spectra.models.base_adapter import ModelAdapter
from vlm_spectra.models.registry import ModelRegistry


@ModelRegistry.register("smolvlm")
class SmolVLMAdapter(ModelAdapter):
    SUPPORTED_MODELS = [
        "HuggingFaceTB/SmolVLM-Instruct",
    ]

    MODEL_CLASS = Idefics3ForConditionalGeneration
    PROCESSOR_CLASS = AutoProcessor

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        # Ensure attention weights are returned (sdpa/fused backends may skip them).
        if hasattr(self.model.model.text_model, "set_attn_implementation"):
            self.model.model.text_model.set_attn_implementation("eager")

    @property
    def lm_layers(self):
        return self.model.model.text_model.layers

    @property
    def lm_attn(self):
        return ModuleList([layer.self_attn for layer in self.lm_layers])

    @property
    def lm_o_proj(self):
        return ModuleList([layer.self_attn.o_proj for layer in self.lm_layers])

    @property
    def lm_mlp(self):
        return ModuleList([layer.mlp for layer in self.lm_layers])

    def get_lm_layer(self, layer_idx: int) -> nn.Module:
        return self.lm_layers[layer_idx]

    def get_lm_attn(self, layer_idx: int) -> nn.Module:
        return self.lm_attn[layer_idx]

    def get_lm_o_proj(self, layer_idx: int) -> nn.Module:
        return self.lm_o_proj[layer_idx]

    def get_lm_mlp(self, layer_idx: int) -> nn.Module:
        return self.lm_mlp[layer_idx]

    def get_lm_norm(self) -> nn.Module:
        return self.model.model.text_model.norm

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def lm_num_layers(self) -> int:
        return self.model.model.text_model.config.num_hidden_layers

    @property
    def lm_num_heads(self) -> int:
        return self.model.model.text_model.config.num_attention_heads

    @property
    def lm_hidden_dim(self) -> int:
        return self.model.model.text_model.config.hidden_size

    @property
    def lm_head_dim(self) -> int:
        return self.lm_hidden_dim // self.lm_num_heads

    def compute_per_head_contributions(
        self, concatenated_heads: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Project each head contribution to the residual stream."""
        o_proj_weight = self.lm_o_proj[layer].weight

        heads_separated = rearrange(
            concatenated_heads,
            "batch seq (heads head_dim) -> batch seq heads head_dim",
            heads=self.lm_num_heads,
        )

        o_proj_per_head = rearrange(
            o_proj_weight,
            "hidden (heads head_dim) -> heads head_dim hidden",
            heads=self.lm_num_heads,
        )

        per_head_contributions = einsum(
            heads_separated,
            o_proj_per_head,
            "batch seq heads head_dim, heads head_dim hidden -> batch seq heads hidden",
        )

        return per_head_contributions

    def compute_attention_patterns(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    ) -> torch.Tensor:
        """Compute attention patterns using the decoder attention."""
        _ = position_ids
        attn_layer = self.lm_attn[layer]

        _, attn_weights = attn_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=True,
        )

        if attn_weights is None:
            raise RuntimeError("Attention weights are None; ensure eager attention is enabled.")

        return attn_weights

    def format_cache_item(self, hook_name: str, cache_item):
        if hook_name in {"lm_resid_pre", "lm.layer.pre"}:
            return self._unwrap_tensor(cache_item)
        if hook_name in {"lm_resid_post", "lm.layer.post"}:
            return self._unwrap_tensor(cache_item)
        if hook_name in {"lm_attn_out", "lm.attn.out"}:
            return cache_item.detach()
        if hook_name in {"lm_mlp_out", "lm.mlp.out"}:
            return cache_item.detach()
        if hook_name in {"lm_attn_pattern", "lm.attn.pattern"}:
            return cache_item.detach()
        return super().format_cache_item(hook_name, cache_item)

    def format_cache(self, cache: dict) -> dict:
        for key, value in cache.items():
            cache[key] = self.format_cache_item(key[0], value)
        return cache

    def get_image_token_id(self) -> int:
        if self.processor is None:
            raise RuntimeError("Processor not set. Call set_processor() first.")
        token_id = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        if token_id != -1:
            return int(token_id)
        if hasattr(self.model.config, "image_token_id"):
            return int(self.model.config.image_token_id)
        raise ValueError("Image token id could not be determined for this model.")

    @staticmethod
    def _unwrap_tensor(cache_item):
        if isinstance(cache_item, tuple):
            return cache_item[0].detach()
        return cache_item.detach()
