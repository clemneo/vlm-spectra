from abc import ABC, abstractmethod
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
from torch.nn.modules import ModuleList
from einops import rearrange, einsum

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