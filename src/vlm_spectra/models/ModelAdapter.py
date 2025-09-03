from abc import ABC, abstractmethod
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
from torch.nn.modules import ModuleList
from einops import rearrange, einsum
import math

# Import Qwen2.5-VL specific functions
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_multimodal_rotary_pos_emb, 
        repeat_kv, 
        rotate_half
    )
except ImportError:
    # Fallback implementations if not available
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors"""
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class ModelAdapter(ABC):
    def __init__(self, model):
        self.model = model
    
    @property
    @abstractmethod
    def lm_num_layers(self) -> int:
        """Return number of language model layers"""
        pass
    
    @property
    @abstractmethod
    def lm_num_heads(self) -> int:
        """Return number of language model heads"""
        pass

    @property
    @abstractmethod
    def lm_layers(self):
        """Return list/dict of language model layers"""
        pass
    
    @property
    @abstractmethod
    def lm_attn(self):
        """Return list/dict of attention modules"""
        pass
    
    @property
    @abstractmethod
    def lm_mlp(self):
        """Return list/dict of MLP modules"""
        pass
    
    @property
    @abstractmethod
    def lm_o_proj(self):
        """Return list/dict of attention output projection modules"""
        pass
    
    @property
    @abstractmethod
    def lm_q_proj(self):
        """Return list/dict of attention query projection modules"""
        pass
    
    @property
    @abstractmethod
    def lm_k_proj(self):
        """Return list/dict of attention key projection modules"""
        pass
    
    @abstractmethod
    def format_cache(self, cache: dict) -> dict:
        """Format raw cache into structured format"""
        pass
    
    @abstractmethod
    def compute_per_head_contributions(self, concatenated_heads: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Compute individual head contributions to residual stream
        
        Args:
            concatenated_heads: [batch, seq_len, num_heads * head_dim] - heads before o_proj
            layer: layer index
            
        Returns:
            [batch, seq_len, num_heads, hidden_size] - per-head contributions to residual stream
        """
        pass


def get_model_adapter(model: nn.Module) -> ModelAdapter:
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        return Qwen2_5_VLModelAdapter(model)
    else:
        raise ValueError(f"Model {model} not supported")

class Qwen2_5_VLModelAdapter(ModelAdapter):
    def __init__(self, model):
        super().__init__(model)

    @property
    def lm_layers(self):
        return self.model.language_model.layers
    
    @property
    def lm_attn(self):
        return ModuleList([
            layer.self_attn for layer in self.model.language_model.layers
        ])
    
    @property
    def lm_o_proj(self):
        return ModuleList([
            layer.self_attn.o_proj for layer in self.model.language_model.layers
        ])
    
    @property
    def lm_q_proj(self):
        return ModuleList([
            layer.self_attn.q_proj for layer in self.model.language_model.layers
        ])
    
    @property
    def lm_k_proj(self):
        return ModuleList([
            layer.self_attn.k_proj for layer in self.model.language_model.layers
        ])
    
    @property
    def lm_mlp(self):
        return ModuleList([
            layer.mlp for layer in self.model.language_model.layers
        ])
    
    @property
    def lm_num_layers(self):
        return self.model.language_model.config.num_hidden_layers
    
    @property
    def lm_num_heads(self):
        return self.model.language_model.config.num_attention_heads
    
    @property
    def lm_hidden_dim(self):
        return self.model.language_model.config.hidden_size

    @property
    def lm_head_dim(self):
        return self.model.language_model.config.hidden_size // self.model.language_model.config.num_attention_heads # not sure if this is correct
    
    @property
    def lm_head(self):
        return self.model.lm_head
    
    @property
    def lm_norm(self):
        return self.model.language_model.norm
    
    def compute_per_head_contributions(self, concatenated_heads: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Compute individual head contributions to residual stream for Qwen2.5-VL
        
        Args:
            concatenated_heads: [batch, seq_len, num_heads * head_dim] - heads before o_proj
            layer: layer index
            
        Returns:
            [batch, seq_len, num_heads, hidden_size] - per-head contributions to residual stream
        """
        # Get the o_proj weight matrix for this layer
        o_proj_weight = self.lm_o_proj[layer].weight  # [hidden_size, num_heads * head_dim]
        
        # Reshape concatenated heads to separate heads
        heads_separated = rearrange(concatenated_heads, 
                                  'batch seq (heads head_dim) -> batch seq heads head_dim',
                                  heads=self.lm_num_heads)
        
        # Reshape o_proj weights to per-head weights
        o_proj_per_head = rearrange(o_proj_weight,
                                  'hidden (heads head_dim) -> heads head_dim hidden',
                                  heads=self.lm_num_heads)
        
        # Compute per-head contributions using einsum
        per_head_contributions = einsum(heads_separated, o_proj_per_head,
                                      'batch seq heads head_dim, heads head_dim hidden -> batch seq heads hidden')
        
        return per_head_contributions
    
    def compute_attention_patterns(self, hidden_states: torch.Tensor, layer: int, attention_mask=None, position_ids=None, position_embeddings=None) -> torch.Tensor:
        """
        Compute attention patterns (attention weights) for a given layer using the exact same logic as Qwen2.5-VL
        
        NOTE: This method can't be fully accurate without position_embeddings and attention_mask.
        For exact matching with output_attentions=True, we need to capture these during the forward pass.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size] - input to the attention layer
            layer: layer index
            attention_mask: attention mask tensor (optional)
            position_ids: position ids tensor (optional) 
            position_embeddings: tuple of (cos, sin) tensors for RoPE (optional)
            
        Returns:
            [batch, num_heads, seq_len, seq_len] - attention patterns (softmax of Q@K^T)
        """
        # Get the attention layer for this specific layer
        attn_layer = self.lm_attn[layer]
        
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V using the layer's projection modules
        query_states = attn_layer.q_proj(hidden_states)
        key_states = attn_layer.k_proj(hidden_states)
        
        # Reshape to separate heads - following exact Qwen2.5-VL logic
        query_states = query_states.view(bsz, q_len, -1, attn_layer.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, attn_layer.head_dim).transpose(1, 2)

        # Apply RoPE if position embeddings are available
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states, key_states, cos, sin, attn_layer.rope_scaling["mrope_section"]
            )

        # Handle grouped query attention - repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, attn_layer.num_key_value_groups)

        # Compute attention weights: Q @ K^T / sqrt(head_dim) - exact Qwen2.5-VL logic
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_layer.head_dim)

        # Apply attention mask - create causal mask if none provided
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        else:
            # Create causal mask for autoregressive generation
            # In Qwen2.5-VL, when attention_mask is None, causal masking is applied
            seq_len = attn_weights.size(-1)
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=attn_weights.device, dtype=attn_weights.dtype), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference (from original implementation)
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # Apply softmax - exact Qwen2.5-VL logic: upcast to fp32, then back
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        return attn_weights
    
    def format_cache(self, cache: dict) -> dict:
        # return cache
        for key, value in cache.items():
            hook_pos, layer = key
            if "resid_pre" in hook_pos:
                cache[key] = self._format_resid_pre(value)
            elif "resid_post" in hook_pos:
                cache[key] = self._format_resid_post(value)
            elif "resid_out" in hook_pos:
                cache[key] = self._format_resid_out(value)
            elif "resid_mid" in hook_pos:
                cache[key] = self._format_resid_mid(value)
            elif "attn_out" in hook_pos:
                cache[key] = self._format_attn_out(value)
            elif "mlp_out" in hook_pos:
                cache[key] = self._format_mlp_out(value)
            elif "attn_pattern" in hook_pos:
                cache[key] = self._format_attn_pattern(value)
            else:
                print(f"Warning: {hook_pos} not supported in cache formatting, skipping")
        return cache

    def _format_resid_pre(self, cache_item: tuple) -> torch.Tensor:
        return cache_item[0].detach()
    
    def _format_resid_post(self, cache_item: tuple) -> torch.Tensor:
        return cache_item[0].detach()
    
    def _format_resid_out(self, cache_item: tuple) -> torch.Tensor:
        raise NotImplementedError("Not implemented")
    
    def _format_resid_mid(self, cache_item: tuple) -> torch.Tensor:
        raise NotImplementedError("Not implemented")
    
    def _format_attn_out(self, cache_item: torch.Tensor) -> torch.Tensor:
        # cache_item is already the per-head contributions tensor [batch, seq, num_heads, hidden_size]
        return cache_item.detach()
    
    
    def _format_mlp_out(self, cache_item: torch.Tensor) -> torch.Tensor:
        # MLP modules return tensors directly, not tuples
        return cache_item.detach()
    
    def _format_attn_pattern(self, cache_item: torch.Tensor) -> torch.Tensor:
        # Attention patterns are computed tensors [batch, num_heads, seq_len, seq_len]
        return cache_item.detach()