"""Patch hook utilities for modifying activations during forward passes.

This module provides helper classes and validation for patch hooks that modify
activations at specific hook points during model execution.
"""

from __future__ import annotations

from typing import Callable, Optional, Set, Union

import torch
import torch.nn as nn

# Type alias for patch hook functions
# Signature: (module, args, kwargs, output) -> modified_output | None
HookFn = Callable[[nn.Module, tuple, dict, torch.Tensor], Optional[torch.Tensor]]

# Valid hook types for patching (is_pre=False AND is_virtual=False)
VALID_PATCH_HOOK_TYPES: Set[str] = {
    "hook_resid_post",
    "attn.hook_q",
    "attn.hook_k",
    "attn.hook_v",
    "attn.hook_out",
    "mlp.hook_pre",
    "mlp.hook_pre_linear",
    "mlp.hook_out",
}

# Valid pre-hook types for patching (is_pre=True but patchable)
VALID_PRE_PATCH_HOOK_TYPES: Set[str] = {
    "hook_resid_pre",
    "attn.hook_in",
    "attn.hook_z",
    "mlp.hook_in",
    "mlp.hook_post",
}


def validate_patch_hook_type(hook_type: str) -> bool:
    """Validate that a hook type can be used for patching.

    Patch hooks can only be registered on hook points that are:
    - Post-hooks (standard patching) OR
    - Specific pre-hooks that support patching (e.g., attn.hook_z for head patching)
    - Not virtual hooks (they require computation, not direct capture)

    Args:
        hook_type: The hook type to validate (e.g., 'hook_resid_post')

    Returns:
        True if this is a pre-hook that supports patching, False for post-hooks

    Raises:
        ValueError: If the hook type is invalid for patching
    """
    if hook_type in VALID_PATCH_HOOK_TYPES:
        return False  # Post-hook

    if hook_type in VALID_PRE_PATCH_HOOK_TYPES:
        return True  # Pre-hook that supports patching

    # Provide helpful error messages for common mistakes
    all_valid = sorted(VALID_PATCH_HOOK_TYPES | VALID_PRE_PATCH_HOOK_TYPES)
    virtual_hooks = {"attn.hook_scores", "attn.hook_pattern", "attn.hook_head_out", "hook_resid_mid"}

    if hook_type in virtual_hooks:
        raise ValueError(
            f"Hook type '{hook_type}' is a virtual hook and cannot be used for patching. "
            f"Virtual hooks are computed from other activations. Valid hook types: {all_valid}"
        )
    else:
        raise ValueError(
            f"Unknown hook type '{hook_type}'. Valid hook types for patching: {all_valid}"
        )


def _select_output_tensor(output: Union[torch.Tensor, tuple]) -> torch.Tensor:
    """Extract the main tensor from module output (handles tuple outputs)."""
    if isinstance(output, tuple):
        return output[0]
    return output


class PatchActivation:
    """Replace activations with a fixed tensor.

    Args:
        replacement: Tensor to replace activations with. Shape depends on token_idx:
            - If token_idx is None: [seq_len, hidden_dim] or [hidden_dim] for single position
            - If token_idx is specified: [hidden_dim]
        token_idx: Optional token position to patch. If None, patches all tokens.
        batch_idx: Batch index to patch (default 0)
    """

    def __init__(
        self,
        replacement: torch.Tensor,
        token_idx: Optional[int] = None,
        batch_idx: int = 0,
    ) -> None:
        self.replacement = replacement
        self.token_idx = token_idx
        self.batch_idx = batch_idx

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        _ = module, args, kwargs
        tensor = _select_output_tensor(output)

        if self.token_idx is None:
            tensor[self.batch_idx] = self.replacement
        else:
            tensor[self.batch_idx, self.token_idx] = self.replacement

        return output


class ZeroAblation:
    """Zero out activations.

    Args:
        token_idx: Optional token position to zero. If None, zeros all tokens.
        batch_idx: Batch index to zero (default 0)
    """

    def __init__(
        self,
        token_idx: Optional[int] = None,
        batch_idx: int = 0,
    ) -> None:
        self.token_idx = token_idx
        self.batch_idx = batch_idx

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        _ = module, args, kwargs
        tensor = _select_output_tensor(output)

        if self.token_idx is None:
            tensor[self.batch_idx] = 0
        else:
            tensor[self.batch_idx, self.token_idx] = 0

        return output


class AddActivation:
    """Add a direction to activations.

    Args:
        direction: Direction tensor to add. Shape should match the activation slice.
        scale: Scaling factor for the direction (default 1.0)
        token_idx: Optional token position. If None, adds to all tokens.
        batch_idx: Batch index to modify (default 0)
    """

    def __init__(
        self,
        direction: torch.Tensor,
        scale: float = 1.0,
        token_idx: Optional[int] = None,
        batch_idx: int = 0,
    ) -> None:
        self.direction = direction
        self.scale = scale
        self.token_idx = token_idx
        self.batch_idx = batch_idx

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        _ = module, args, kwargs
        tensor = _select_output_tensor(output)

        direction = self.direction.to(tensor.device, tensor.dtype)
        if self.token_idx is None:
            tensor[self.batch_idx] = tensor[self.batch_idx] + self.scale * direction
        else:
            tensor[self.batch_idx, self.token_idx] = (
                tensor[self.batch_idx, self.token_idx] + self.scale * direction
            )

        return output


class ScaleActivation:
    """Scale activations by a factor.

    Args:
        scale: Scaling factor
        token_idx: Optional token position. If None, scales all tokens.
        batch_idx: Batch index to scale (default 0)
    """

    def __init__(
        self,
        scale: float,
        token_idx: Optional[int] = None,
        batch_idx: int = 0,
    ) -> None:
        self.scale = scale
        self.token_idx = token_idx
        self.batch_idx = batch_idx

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        _ = module, args, kwargs
        tensor = _select_output_tensor(output)

        if self.token_idx is None:
            tensor[self.batch_idx] = tensor[self.batch_idx] * self.scale
        else:
            tensor[self.batch_idx, self.token_idx] = (
                tensor[self.batch_idx, self.token_idx] * self.scale
            )

        return output


class PatchHead:
    """Patch a specific attention head's output.

    This is designed for use with attn.hook_z (pre o_proj) where the tensor
    has shape [batch, seq, num_heads * head_dim]. The tensor is reshaped to
    [batch, seq, num_heads, head_dim], patched, then reshaped back.

    Example:
        # Use adapter properties for num_heads and head_dim
        num_heads = model.adapter.lm_num_heads
        head_dim = model.adapter.lm_head_dim
        replacement = torch.zeros(head_dim, device=model.device)

        patch = PatchHead(head_idx=0, replacement=replacement, num_heads=num_heads)
        with model.run_with_hooks([("lm.blocks.5.attn.hook_z", patch)]):
            model.forward(inputs)

    Args:
        head_idx: Index of the head to patch
        replacement: Replacement tensor of shape [head_dim] or [seq_len, head_dim]
        num_heads: Number of attention heads in the model
        token_idx: Optional token position. If None, patches all tokens.
        batch_idx: Batch index to patch (default 0)
    """

    def __init__(
        self,
        head_idx: int,
        replacement: torch.Tensor,
        num_heads: int,
        token_idx: Optional[int] = None,
        batch_idx: int = 0,
    ) -> None:
        self.head_idx = head_idx
        self.replacement = replacement
        self.num_heads = num_heads
        self.token_idx = token_idx
        self.batch_idx = batch_idx

    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: Union[torch.Tensor, tuple],
    ) -> Union[torch.Tensor, tuple]:
        _ = module, args, kwargs
        tensor = _select_output_tensor(output)

        batch, seq, hidden = tensor.shape
        head_dim = hidden // self.num_heads

        # Reshape to [batch, seq, num_heads, head_dim]
        heads_separated = tensor.view(batch, seq, self.num_heads, head_dim)

        replacement = self.replacement.to(tensor.device, tensor.dtype)
        if self.token_idx is None:
            heads_separated[self.batch_idx, :, self.head_idx] = replacement
        else:
            heads_separated[self.batch_idx, self.token_idx, self.head_idx] = replacement

        # Reshape back - this modifies tensor in place since view shares memory
        tensor.view(batch, seq, self.num_heads, head_dim)

        return output
