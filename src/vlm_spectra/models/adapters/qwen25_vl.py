from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn.modules import ModuleList
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vlm_spectra.models.base_adapter import ModelAdapter
from vlm_spectra.models.registry import ModelRegistry

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb,
        repeat_kv,
        rotate_half,
    )
except ImportError:

    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads to match attention heads."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        """Applies RoPE with multimodal sections to query/key tensors."""
        mrope_section = mrope_section * 2
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))],
            dim=-1,
        ).unsqueeze(unsqueeze_dim)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


@ModelRegistry.register("qwen25-vl")
class Qwen25VLAdapter(ModelAdapter):
    SUPPORTED_MODELS = [
        "ByteDance-Seed/UI-TARS-1.5-7B",
    ]

    MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
    PROCESSOR_CLASS = AutoProcessor

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")

    @property
    def lm_layers(self):
        return self.model.language_model.layers

    @property
    def lm_attn(self):
        return ModuleList([layer.self_attn for layer in self.model.language_model.layers])

    @property
    def lm_o_proj(self):
        return ModuleList(
            [layer.self_attn.o_proj for layer in self.model.language_model.layers]
        )

    @property
    def lm_q_proj(self):
        return ModuleList(
            [layer.self_attn.q_proj for layer in self.model.language_model.layers]
        )

    @property
    def lm_k_proj(self):
        return ModuleList(
            [layer.self_attn.k_proj for layer in self.model.language_model.layers]
        )

    @property
    def lm_mlp(self):
        return ModuleList([layer.mlp for layer in self.model.language_model.layers])

    def get_lm_layer(self, layer_idx: int) -> nn.Module:
        return self.lm_layers[layer_idx]

    def get_lm_attn(self, layer_idx: int) -> nn.Module:
        return self.lm_attn[layer_idx]

    def get_lm_o_proj(self, layer_idx: int) -> nn.Module:
        return self.lm_o_proj[layer_idx]

    def get_lm_mlp(self, layer_idx: int) -> nn.Module:
        return self.lm_mlp[layer_idx]

    def get_lm_norm(self) -> nn.Module:
        return self.model.language_model.norm

    def get_lm_head(self) -> nn.Module:
        return self.model.lm_head

    @property
    def lm_num_layers(self) -> int:
        return self.model.language_model.config.num_hidden_layers

    @property
    def lm_num_heads(self) -> int:
        return self.model.language_model.config.num_attention_heads

    @property
    def lm_hidden_dim(self) -> int:
        return self.model.language_model.config.hidden_size

    @property
    def lm_head_dim(self) -> int:
        return (
            self.model.language_model.config.hidden_size
            // self.model.language_model.config.num_attention_heads
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def lm_norm(self):
        return self.model.language_model.norm

    def compute_per_head_contributions(
        self, concatenated_heads: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Compute individual head contributions to the residual stream."""
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
        """Compute attention patterns using Qwen2.5-VL attention logic."""
        _ = position_ids
        attn_layer = self.lm_attn[layer]

        bsz, q_len, _ = hidden_states.size()

        query_states = attn_layer.q_proj(hidden_states)
        key_states = attn_layer.k_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, attn_layer.head_dim).transpose(
            1, 2
        )
        key_states = key_states.view(bsz, q_len, -1, attn_layer.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                attn_layer.rope_scaling["mrope_section"],
            )

        key_states = repeat_kv(key_states, attn_layer.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(attn_layer.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        else:
            seq_len = attn_weights.size(-1)
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len),
                    float("-inf"),
                    device=attn_weights.device,
                    dtype=attn_weights.dtype,
                ),
                diagonal=1,
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights + causal_mask

        if query_states.dtype == torch.float16:
            attn_weights = torch.where(
                torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights
            )

        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

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
        return self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    @staticmethod
    def _unwrap_tensor(cache_item):
        if isinstance(cache_item, tuple):
            return cache_item[0].detach()
        return cache_item.detach()
