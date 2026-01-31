from __future__ import annotations

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn.modules import ModuleList
from transformers import AutoProcessor

try:
    from transformers import Idefics3ForConditionalGeneration
except ImportError:
    Idefics3ForConditionalGeneration = None

if Idefics3ForConditionalGeneration is not None:
    from vlm_spectra.models.base_adapter import ModelAdapter
    from vlm_spectra.models.registry import ModelRegistry

    @ModelRegistry.register("smolvlm")
    class SmolVLMAdapter(ModelAdapter):
        SUPPORTED_MODELS = [
            "HuggingFaceTB/SmolVLM-Instruct",
            "HuggingFaceTB/SmolVLM-256M-Instruct",
            "HuggingFaceTB/SmolVLM-500M-Instruct",
        ]

        MODEL_CLASS = Idefics3ForConditionalGeneration
        PROCESSOR_CLASS = AutoProcessor

        def __init__(self, model: nn.Module) -> None:
            super().__init__(model)
            self._text_model = model.model.text_model
            # TODO: Only set eager attention when output_attentions=True is requested,
            # then restore the original implementation afterwards. Currently we always
            # use eager which is slower but supports attention output.
            if hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation("eager")

        @property
        def lm_layers(self):
            return self._text_model.layers

        @property
        def lm_attn(self):
            return ModuleList([layer.self_attn for layer in self._text_model.layers])

        @property
        def lm_o_proj(self):
            return ModuleList(
                [layer.self_attn.o_proj for layer in self._text_model.layers]
            )

        @property
        def lm_mlp(self):
            return ModuleList([layer.mlp for layer in self._text_model.layers])

        def get_lm_layer(self, layer_idx: int) -> nn.Module:
            return self.lm_layers[layer_idx]

        def get_lm_attn(self, layer_idx: int) -> nn.Module:
            return self.lm_attn[layer_idx]

        def get_lm_o_proj(self, layer_idx: int) -> nn.Module:
            return self.lm_o_proj[layer_idx]

        def get_lm_mlp(self, layer_idx: int) -> nn.Module:
            return self.lm_mlp[layer_idx]

        def get_lm_q_proj(self, layer_idx: int) -> nn.Module:
            return self._text_model.layers[layer_idx].self_attn.q_proj

        def get_lm_k_proj(self, layer_idx: int) -> nn.Module:
            return self._text_model.layers[layer_idx].self_attn.k_proj

        def get_lm_v_proj(self, layer_idx: int) -> nn.Module:
            return self._text_model.layers[layer_idx].self_attn.v_proj

        def get_lm_norm(self) -> nn.Module:
            return self._text_model.norm

        def get_lm_head(self) -> nn.Module:
            return self.model.lm_head

        @property
        def lm_num_layers(self) -> int:
            return self._text_model.config.num_hidden_layers

        @property
        def lm_num_heads(self) -> int:
            return self._text_model.config.num_attention_heads

        @property
        def lm_hidden_dim(self) -> int:
            return self._text_model.config.hidden_size

        @property
        def lm_head_dim(self) -> int:
            return (
                self._text_model.config.hidden_size
                // self._text_model.config.num_attention_heads
            )

        @property
        def lm_num_kv_heads(self) -> int:
            return self._text_model.config.num_key_value_heads

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
            """Compute attention patterns using Llama attention with output_attentions."""
            attn_layer = self.lm_attn[layer]

            kwargs = {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
                "output_attentions": True,
            }

            if position_embeddings is not None:
                kwargs["position_embeddings"] = position_embeddings
            elif position_ids is not None:
                kwargs["position_ids"] = position_ids

            attn_outputs = attn_layer(**kwargs)
            return attn_outputs[1]

        def get_image_token_id(self) -> int:
            if self.processor is None:
                raise RuntimeError("Processor not set. Call set_processor() first.")
            return self.processor.tokenizer.convert_tokens_to_ids("<image>")

        def format_cache_item(self, hook_type: str, cache_item):
            """Format a cache item based on hook type.

            Args:
                hook_type: Hook type like 'hook_resid_post' or 'attn.hook_pattern'
                cache_item: The cached tensor or tuple
            """
            # Residual stream hooks may return tuples
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
                raise NotImplementedError("attn.hook_head_out not implemented for SmolVLM")
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
            if hook_type == "attn.hook_pattern":
                return cache_item.detach()
            return super().format_cache_item(hook_type, cache_item)

        @staticmethod
        def _unwrap_tensor(cache_item):
            if isinstance(cache_item, tuple):
                return cache_item[0].detach()
            return cache_item.detach()
