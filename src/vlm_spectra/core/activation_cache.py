from __future__ import annotations

import re
from typing import Dict, Iterator, List

import torch

from vlm_spectra.core.hook_points import HookPoint


class ActivationCache:
    """Cache for model activations with string keys.

    Keys follow TransformerLens naming convention:
        lm.blocks.{layer}.{hook_type}

    Examples:
        cache["lm.blocks.5.hook_resid_post"]
        cache.stack("lm.blocks.*.hook_resid_post")
    """

    def __init__(self) -> None:
        self._data: Dict[str, torch.Tensor] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get activation by key.

        Args:
            key: Hook name like 'lm.blocks.5.hook_resid_post'
        """
        return self._data[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Set activation by key."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def values(self) -> List[torch.Tensor]:
        return list(self._data.values())

    def items(self):
        return self._data.items()

    def stack(self, pattern: str) -> torch.Tensor:
        """Stack all activations matching a pattern.

        Args:
            pattern: Pattern with wildcard, e.g., 'lm.blocks.*.hook_resid_post'

        Returns:
            Stacked tensor with shape [num_matches, ...]

        Example:
            >>> stacked = cache.stack("lm.blocks.*.hook_resid_post")
            >>> stacked.shape
            torch.Size([28, 1, 512, 3584])  # [num_layers, batch, seq, hidden]
        """
        matching_keys = self._match_pattern(pattern)
        if not matching_keys:
            raise KeyError(f"No keys match pattern: {pattern}")

        # Sort by layer number to ensure consistent ordering
        def get_layer(key: str) -> int:
            _, layer = HookPoint.parse(key)
            return layer if isinstance(layer, int) else 0

        sorted_keys = sorted(matching_keys, key=get_layer)
        return torch.stack([self._data[k] for k in sorted_keys])

    def _match_pattern(self, pattern: str) -> List[str]:
        """Find all keys matching a pattern with wildcards."""
        if "*" not in pattern:
            return [pattern] if pattern in self._data else []

        # Convert pattern to regex: lm.blocks.*.hook_resid_post -> lm\.blocks\.\d+\.hook_resid_post
        regex_pattern = re.escape(pattern).replace(r"\*", r"\d+")
        regex = re.compile(f"^{regex_pattern}$")
        return [k for k in self._data.keys() if regex.match(k)]

    def clear(self) -> None:
        """Clear all cached activations."""
        self._data.clear()

    def detach(self) -> None:
        """Detach all tensors from the computation graph."""
        for key in list(self._data.keys()):
            self._data[key] = self._data[key].detach()
