from __future__ import annotations

from typing import Optional

import torch

from vlm_spectra.hooks.base import Hook


def _select_output_tensor(output):
    if isinstance(output, tuple):
        return output[0]
    return output


class PatchResidualHook(Hook):
    """Replace residual stream at a specific token position."""

    hook_point = "lm.layer.post"

    def __init__(self, layer: int, token_idx: int, replacement: torch.Tensor) -> None:
        self.layer = layer
        self.token_idx = token_idx
        self.replacement = replacement

    def __call__(self, module, args, kwargs, output):
        _ = module
        _ = args
        _ = kwargs
        resid = _select_output_tensor(output)
        resid[0, self.token_idx] = self.replacement
        return output


class PatchHeadHook(Hook):
    """Replace attention head output before o_proj projection."""

    hook_point = "lm.attn.head"

    def __init__(
        self,
        layer: int,
        head_idx: int,
        replacement: torch.Tensor,
        token_idx: Optional[int] = None,
    ) -> None:
        self.layer = layer
        self.head_idx = head_idx
        self.replacement = replacement
        self.token_idx = token_idx
        self._num_heads: Optional[int] = None

    def set_num_heads(self, num_heads: int) -> None:
        """Called by hook manager to inject adapter's num_heads."""
        self._num_heads = num_heads

    def __call__(self, module, args, kwargs, output):
        """Modify concatenated heads before o_proj.

        Args shape: [batch, seq, num_heads * head_dim]
        We reshape to [batch, seq, num_heads, head_dim], patch, reshape back.
        """
        _ = module
        _ = output  # None for pre-hooks

        if self._num_heads is None:
            raise RuntimeError(
                "num_heads not set. Hook must be registered via HookManager."
            )

        # Get input tensor
        if len(args) > 0:
            concat_heads = args[0]
        else:
            concat_heads = kwargs["hidden_states"]

        batch, seq, hidden = concat_heads.shape
        head_dim = hidden // self._num_heads

        # Reshape to [batch, seq, heads, head_dim]
        heads_separated = concat_heads.view(batch, seq, self._num_heads, head_dim)

        # Patch the specific head
        if self.token_idx is None:
            # replacement shape: [seq, head_dim]
            heads_separated[0, :, self.head_idx] = self.replacement
        else:
            # replacement shape: [head_dim]
            heads_separated[0, self.token_idx, self.head_idx] = self.replacement

        # Reshape back to [batch, seq, hidden]
        modified = heads_separated.view(batch, seq, hidden)

        # Return modified args for pre-hook
        if len(args) > 0:
            return (modified,) + args[1:], kwargs
        else:
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = modified
            return args, new_kwargs


class PatchMLPHook(Hook):
    """Replace MLP output at specific token position."""

    hook_point = "lm.mlp.out"

    def __init__(self, layer: int, replacement: torch.Tensor, token_idx: int) -> None:
        self.layer = layer
        self.replacement = replacement
        self.token_idx = token_idx

    def __call__(self, module, args, kwargs, output):
        _ = module
        _ = args
        _ = kwargs
        mlp_out = _select_output_tensor(output)
        mlp_out[0, self.token_idx] = self.replacement
        return output
