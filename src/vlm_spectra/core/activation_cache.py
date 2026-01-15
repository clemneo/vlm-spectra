from __future__ import annotations

from typing import Dict, List, Tuple

import torch


class ActivationCache:
    """Cache for model activations with intuitive access."""

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, int], torch.Tensor] = {}

    def __getitem__(self, key: Tuple[str, int]) -> torch.Tensor:
        """Support cache["lm.attn.out", 5] style access."""
        return self._data[key]

    def __setitem__(self, key: Tuple[str, int], value: torch.Tensor) -> None:
        self._data[key] = value

    def __contains__(self, key: Tuple[str, int]) -> bool:
        return key in self._data

    def keys(self) -> List[Tuple[str, int]]:
        return list(self._data.keys())

    def items(self):
        return self._data.items()

    def get_all_layers(self, name: str) -> Dict[int, torch.Tensor]:
        """Get all layers for a hook name as a dict."""
        return {
            layer: tensor
            for (hook_name, layer), tensor in self._data.items()
            if hook_name == name
        }

    def stack_layers(self, name: str) -> torch.Tensor:
        """Stack all layers for a hook name into [num_layers, ...]."""
        layers = self.get_all_layers(name)
        return torch.stack([layers[i] for i in sorted(layers.keys())])

    def clear(self) -> None:
        self._data.clear()

    def detach(self) -> None:
        """Detach all tensors from the computation graph."""
        for key in list(self._data.keys()):
            self._data[key] = self._data[key].detach()
