from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """Abstract adapter for model-specific internals."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.processor = None

    def set_processor(self, processor) -> None:
        """Set the processor for this adapter."""
        self.processor = processor

    @property
    @abstractmethod
    def lm_num_layers(self) -> int:
        """Return number of language model layers."""

    @property
    @abstractmethod
    def lm_num_heads(self) -> int:
        """Return number of language model heads."""

    @property
    @abstractmethod
    def lm_hidden_dim(self) -> int:
        """Return language model hidden dimension."""

    @property
    @abstractmethod
    def lm_head_dim(self) -> int:
        """Return per-head hidden dimension."""

    @abstractmethod
    def get_lm_layer(self, layer_idx: int) -> nn.Module:
        """Return the language model layer at index."""

    @abstractmethod
    def get_lm_attn(self, layer_idx: int) -> nn.Module:
        """Return the attention module at index."""

    @abstractmethod
    def get_lm_o_proj(self, layer_idx: int) -> nn.Module:
        """Return the attention output projection module at index."""

    @abstractmethod
    def get_lm_mlp(self, layer_idx: int) -> nn.Module:
        """Return the MLP module at index."""

    # QKV projection getters (required for attn.hook_q/k/v)
    @abstractmethod
    def get_lm_q_proj(self, layer_idx: int) -> nn.Module:
        """Return the query projection module at index."""

    @abstractmethod
    def get_lm_k_proj(self, layer_idx: int) -> nn.Module:
        """Return the key projection module at index."""

    @abstractmethod
    def get_lm_v_proj(self, layer_idx: int) -> nn.Module:
        """Return the value projection module at index."""

    # MLP internal getters (optional, for mlp.hook_pre/pre_linear/post)
    def get_lm_gate_proj(self, layer_idx: int) -> nn.Module:
        """Return the MLP gate projection (for gated MLPs like SwiGLU)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_lm_gate_proj"
        )

    def get_lm_up_proj(self, layer_idx: int) -> nn.Module:
        """Return the MLP up projection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_lm_up_proj"
        )

    def get_lm_down_proj(self, layer_idx: int) -> nn.Module:
        """Return the MLP down projection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_lm_down_proj"
        )

    def get_lm_norm(self) -> nn.Module:
        """Return the final normalization layer if available."""
        raise NotImplementedError

    def get_lm_head(self) -> nn.Module:
        """Return the LM head if available."""
        raise NotImplementedError

    @abstractmethod
    def compute_per_head_contributions(
        self, concatenated_heads: torch.Tensor, layer: int
    ) -> torch.Tensor:
        """Compute per-head contributions to the residual stream."""

    @abstractmethod
    def compute_attention_patterns(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
    ) -> torch.Tensor:
        """Compute attention patterns for a given layer."""

    def format_cache_item(self, hook_type: str, cache_item):
        """Format a single cache item to a tensor.

        Args:
            hook_type: Hook type like 'hook_resid_post' or 'attn.hook_pattern'
            cache_item: The cached tensor or tuple
        """
        _ = hook_type
        if isinstance(cache_item, torch.Tensor):
            return cache_item.detach()
        return cache_item

    def format_cache(self, cache: dict) -> dict:
        """Format raw cache into structured format."""
        for key, value in cache.items():
            cache[key] = self.format_cache_item(key[0], value)
        return cache

    @abstractmethod
    def get_image_token_id(self) -> int:
        """Get the token ID used for image patches in this model."""

    @property
    def has_strict_residual_stream(self) -> bool:
        """Whether hook_resid_post equals next hook_resid_pre exactly."""

        return True
