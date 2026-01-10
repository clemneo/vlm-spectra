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
    """Replace attention head output."""

    hook_point = "lm.attn.out"

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

    def __call__(self, module, args, kwargs, output):
        _ = module
        _ = args
        _ = kwargs
        attn_out = _select_output_tensor(output)
        if self.token_idx is None:
            attn_out[0, :, self.head_idx] = self.replacement
        else:
            attn_out[0, self.token_idx, self.head_idx] = self.replacement
        return output


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
