from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn.modules import ModuleList
from transformers import AutoProcessor
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    LlavaForConditionalGeneration = None

if LlavaForConditionalGeneration is not None:
    from vlm_spectra.models.base_adapter import ModelAdapter
    from vlm_spectra.models.registry import ModelRegistry

    @ModelRegistry.register("llava")
    class LlavaAdapter(ModelAdapter):
        SUPPORTED_MODELS = [
            "llava-hf/llava-1.5-7b-hf",
            "llava-hf/llava-1.5-13b-hf",
        ]

        MODEL_CLASS = LlavaForConditionalGeneration
        PROCESSOR_CLASS = AutoProcessor

        def __init__(self, model: nn.Module) -> None:
            super().__init__(model)
            self._language_model = model.language_model

        @property
        def lm_layers(self):
            return self._language_model.layers

        @property
        def lm_attn(self):
            return ModuleList(
                [layer.self_attn for layer in self._language_model.layers]
            )

        @property
        def lm_o_proj(self):
            return ModuleList(
                [layer.self_attn.o_proj for layer in self._language_model.layers]
            )

        @property
        def lm_mlp(self):
            return ModuleList([layer.mlp for layer in self._language_model.layers])

        def get_lm_layer(self, layer_idx: int) -> nn.Module:
            return self.lm_layers[layer_idx]

        def get_lm_attn(self, layer_idx: int) -> nn.Module:
            return self.lm_attn[layer_idx]

        def get_lm_o_proj(self, layer_idx: int) -> nn.Module:
            return self.lm_o_proj[layer_idx]

        def get_lm_mlp(self, layer_idx: int) -> nn.Module:
            return self.lm_mlp[layer_idx]

        def get_lm_q_proj(self, layer_idx: int) -> nn.Module:
            return self._language_model.layers[layer_idx].self_attn.q_proj

        def get_lm_k_proj(self, layer_idx: int) -> nn.Module:
            return self._language_model.layers[layer_idx].self_attn.k_proj

        def get_lm_v_proj(self, layer_idx: int) -> nn.Module:
            return self._language_model.layers[layer_idx].self_attn.v_proj

        def get_lm_gate_proj(self, layer_idx: int) -> nn.Module:
            return self._language_model.layers[layer_idx].mlp.gate_proj

        def get_lm_up_proj(self, layer_idx: int) -> nn.Module:
            return self._language_model.layers[layer_idx].mlp.up_proj

        def get_lm_down_proj(self, layer_idx: int) -> nn.Module:
            return self._language_model.layers[layer_idx].mlp.down_proj

        def get_lm_norm(self) -> nn.Module:
            return self._language_model.norm

        def get_lm_head(self) -> nn.Module:
            return self.model.lm_head

        @property
        def lm_num_layers(self) -> int:
            return self._language_model.config.num_hidden_layers

        @property
        def lm_num_heads(self) -> int:
            return self._language_model.config.num_attention_heads

        @property
        def lm_hidden_dim(self) -> int:
            return self._language_model.config.hidden_size

        @property
        def lm_head_dim(self) -> int:
            return (
                self._language_model.config.hidden_size
                // self._language_model.config.num_attention_heads
            )

        @property
        def lm_num_kv_heads(self) -> int:
            return self._language_model.config.num_key_value_heads

        @property
        def lm_mlp_dim(self) -> int:
            return self._language_model.config.intermediate_size

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
            position_embeddings=None,
        ) -> torch.Tensor:
            """Compute attention patterns manually (matches eager_attention_forward)."""
            attn_layer = self.lm_attn[layer]
            bsz, seq_len, _ = hidden_states.shape
            hidden_shape = (bsz, seq_len, -1, self.lm_head_dim)

            # Q, K projections + reshape (matches LlamaAttention.forward)
            query_states = (
                attn_layer.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            key_states = (
                attn_layer.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )

            # Apply RoPE if position_embeddings provided
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            # GQA expansion
            key_states = repeat_kv(key_states, attn_layer.num_key_value_groups)

            # Pre-softmax scores
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * attn_layer.scaling
            )

            # Apply causal mask
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

            # Softmax in float32 then cast back
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            return attn_weights

        def compute_attention_scores(
            self,
            hidden_states: torch.Tensor,
            layer: int,
            attention_mask=None,
            position_embeddings=None,
        ) -> torch.Tensor:
            """Compute pre-softmax attention scores (matches LlamaAttention.forward)."""
            attn_layer = self.lm_attn[layer]

            bsz, seq_len, _ = hidden_states.shape
            hidden_shape = (bsz, seq_len, -1, self.lm_head_dim)

            query_states = (
                attn_layer.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            key_states = (
                attn_layer.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )

            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            key_states = repeat_kv(key_states, attn_layer.num_key_value_groups)

            scaling = attn_layer.scaling
            attn_scores = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
            )

            return attn_scores

        def get_image_token_id(self) -> int:
            return self.model.config.image_token_index

        def format_cache_item(self, hook_type: str, cache_item):
            """Format a cache item based on hook type."""
            if hook_type in {"hook_resid_pre", "hook_resid_post"}:
                return self._unwrap_tensor(cache_item)
            if hook_type == "attn.hook_out":
                return cache_item.detach()
            if hook_type == "attn.hook_q":
                q = self._unwrap_tensor(cache_item)
                bsz, seq_len, proj_dim = q.shape
                head_dim = self.lm_head_dim
                expected_heads = self.lm_num_heads
                inferred_heads = proj_dim // head_dim
                num_heads = (
                    expected_heads
                    if expected_heads * head_dim == proj_dim
                    else inferred_heads
                )
                return q.reshape(bsz, seq_len, num_heads, head_dim)

            if hook_type == "attn.hook_head_out":
                return cache_item.detach()
            if hook_type == "attn.hook_scores":
                return cache_item.detach()
            if hook_type in {"attn.hook_k", "attn.hook_v"}:
                kv = self._unwrap_tensor(cache_item)
                bsz, seq_len, proj_dim = kv.shape
                head_dim = self.lm_head_dim
                num_kv_heads = proj_dim // head_dim
                return kv.reshape(bsz, seq_len, num_kv_heads, head_dim)
            if hook_type == "attn.hook_z":
                z = self._unwrap_tensor(cache_item)
                bsz, seq_len, proj_dim = z.shape
                head_dim = self.lm_head_dim
                num_heads = self.lm_num_heads
                return z.reshape(bsz, seq_len, num_heads, head_dim)
            if hook_type in {"mlp.hook_out", "mlp.hook_in"}:
                return cache_item.detach()
            if hook_type in {"mlp.hook_pre", "mlp.hook_pre_linear", "mlp.hook_post"}:
                return cache_item.detach()
            if hook_type == "attn.hook_pattern":
                return cache_item.detach()
            return super().format_cache_item(hook_type, cache_item)

        @staticmethod
        def _unwrap_tensor(cache_item):
            if isinstance(cache_item, tuple):
                return cache_item[0].detach()
            return cache_item.detach()
