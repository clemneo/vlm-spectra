from __future__ import annotations

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.hooks.base import Hook


class CaptureOutputHook(Hook):
    """Capture module output."""

    def __init__(self, cache: ActivationCache, hook_name: str, layer: int) -> None:
        self.cache = cache
        self.hook_name = hook_name
        self.layer = layer

    def __call__(self, module, args, kwargs, output):
        _ = module
        _ = args
        _ = kwargs
        self.cache[(self.hook_name, self.layer)] = output
        return output


class CaptureInputHook(Hook):
    """Capture module input (hidden_states from args/kwargs)."""

    def __init__(self, cache: ActivationCache, hook_name: str, layer: int) -> None:
        self.cache = cache
        self.hook_name = hook_name
        self.layer = layer

    def __call__(self, module, args, kwargs, output):
        _ = module
        _ = output
        if len(args) > 0:
            hidden_states = args[0]
        else:
            hidden_states = kwargs.get("hidden_states")
        if hidden_states is not None:
            self.cache[(self.hook_name, self.layer)] = hidden_states
        return output
