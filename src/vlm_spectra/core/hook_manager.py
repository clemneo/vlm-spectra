from __future__ import annotations

from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from vlm_spectra.core.activation_cache import ActivationCache

if TYPE_CHECKING:
    from vlm_spectra.hooks.base import Hook


class HookPointRegistry:
    """Single source of truth for hook point configuration."""

    HOOK_POINTS = {
        "lm.layer.pre": {"module_getter": "get_lm_layer", "is_pre_hook": True},
        "lm.layer.post": {"module_getter": "get_lm_layer", "is_pre_hook": False},
        "lm.attn.out": {"module_getter": "get_lm_o_proj", "is_pre_hook": False},
        "lm.attn.pre": {"module_getter": "get_lm_attn", "is_pre_hook": True},
        "lm.attn.pattern": {"module_getter": "get_lm_attn", "is_pre_hook": True},
        "lm.mlp.out": {"module_getter": "get_lm_mlp", "is_pre_hook": False},
    }

    LEGACY_TO_CANONICAL = {
        "lm_resid_pre": "lm.layer.pre",
        "lm_resid_post": "lm.layer.post",
        "lm_resid_mid": "lm.layer.mid",
        "lm_attn_out": "lm.attn.out",
        "lm_attn_pattern": "lm.attn.pattern",
        "lm_mlp_out": "lm.mlp.out",
    }

    @classmethod
    def canonicalize(cls, name: str) -> str:
        """Convert legacy or canonical name to canonical form."""
        if name in cls.LEGACY_TO_CANONICAL:
            return cls.LEGACY_TO_CANONICAL[name]
        if name in cls.HOOK_POINTS:
            return name
        raise NotImplementedError(f"Unknown hook point: {name}")

    @classmethod
    def get_module(cls, adapter, hook_point: str, layer: int) -> nn.Module:
        """Get the module for a hook point from the adapter."""
        canonical = cls.canonicalize(hook_point)
        config = cls.HOOK_POINTS[canonical]
        getter = getattr(adapter, config["module_getter"])
        return getter(layer)

    @classmethod
    def is_pre_hook(cls, hook_point: str) -> bool:
        """Check if hook point requires a pre-hook."""
        canonical = cls.canonicalize(hook_point)
        return cls.HOOK_POINTS[canonical]["is_pre_hook"]


class HookManager:
    """Manages PyTorch hook lifecycle for interpretability."""

    def __init__(self, adapter: "ModelAdapter") -> None:
        self._adapter = adapter
        self._cache_handles: List[RemovableHandle] = []
        self._patch_handles: List[RemovableHandle] = []
        self._input_cache: Dict[Tuple[str, int], Any] = {}
        self._canonical_names: Dict[str, str] = {}

    def register_cache_hooks(self, cache: ActivationCache, names: List[str]) -> None:
        """Register hooks that capture activations into cache."""
        if self._cache_handles:
            self.remove_cache_hooks()
        self._input_cache = {}
        self._canonical_names = {}

        for name in names:
            if not name.startswith("lm"):
                raise NotImplementedError("Only LM hooks are supported for now")

            canonical = HookPointRegistry.canonicalize(name)
            self._canonical_names[name] = canonical
            if canonical == "lm.layer.mid":
                raise NotImplementedError("Resid_mid hooks are not supported for now")

            is_pre = HookPointRegistry.is_pre_hook(canonical)

            for layer in range(self._adapter.lm_num_layers):
                module = HookPointRegistry.get_module(self._adapter, canonical, layer)

                # Determine hook function and type based on what we're capturing
                if canonical == "lm.attn.out":
                    # lm.attn.out: post-hook on o_proj, capture input for per-head decomposition
                    hook_fn = self._save_input_hook_post(layer, name)
                    handle = module.register_forward_hook(hook_fn, with_kwargs=True)
                elif canonical == "lm.attn.pattern":
                    # lm.attn.pattern: pre-hook on attn, capture inputs for pattern computation
                    hook_fn = self._save_input_hook_pre(layer, name, canonical)
                    handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
                elif is_pre:
                    # Pre-hooks capture input to the module
                    hook_fn = self._save_pre_hook(cache, layer, name)
                    handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
                else:
                    # Post-hooks capture output from the module
                    hook_fn = self._save_output_hook(cache, layer, name)
                    handle = module.register_forward_hook(hook_fn, with_kwargs=True)

                self._cache_handles.append(handle)

    def register_patch_hooks(self, hooks: List["Hook"]) -> None:
        """Register hooks that modify activations."""
        if self._patch_handles:
            self.remove_patch_hooks()
        for hook in hooks:
            hook_point = getattr(hook, "hook_point", "lm.layer.post")
            layer = hook.layer

            module = HookPointRegistry.get_module(self._adapter, hook_point, layer)
            is_pre = HookPointRegistry.is_pre_hook(hook_point)

            if is_pre:
                handle = module.register_forward_pre_hook(
                    self._wrap_pre_hook(hook), with_kwargs=True
                )
            else:
                handle = module.register_forward_hook(hook, with_kwargs=True)
            self._patch_handles.append(handle)

    def finalize_cache(self, cache: ActivationCache, names: List[str]) -> None:
        """Compute derived cache entries that require extra inputs."""
        for name in names:
            canonical = self._canonical_names.get(name, HookPointRegistry.canonicalize(name))
            if canonical == "lm.attn.out":
                for layer in range(self._adapter.lm_num_layers):
                    key = (name, layer)
                    if key in self._input_cache:
                        concatenated_heads = self._input_cache[key]
                        cache[key] = self._adapter.compute_per_head_contributions(
                            concatenated_heads, layer
                        )
            elif canonical == "lm.attn.pattern":
                for layer in range(self._adapter.lm_num_layers):
                    key = (name, layer)
                    if key in self._input_cache:
                        hook_data = self._input_cache[key]
                        if isinstance(hook_data, dict):
                            attn_patterns = self._adapter.compute_attention_patterns(
                                hidden_states=hook_data["hidden_states"],
                                layer=layer,
                                attention_mask=hook_data.get("attention_mask"),
                                position_ids=hook_data.get("position_ids"),
                                position_embeddings=hook_data.get(
                                    "position_embeddings"
                                ),
                            )
                        else:
                            attn_patterns = self._adapter.compute_attention_patterns(
                                hook_data, layer
                            )
                        cache[key] = attn_patterns

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        self.remove_cache_hooks()
        self.remove_patch_hooks()

    def remove_cache_hooks(self) -> None:
        """Remove cache hooks only."""
        for handle in self._cache_handles:
            handle.remove()
        self._cache_handles.clear()

    def remove_patch_hooks(self) -> None:
        """Remove patch hooks only."""
        for handle in self._patch_handles:
            handle.remove()
        self._patch_handles.clear()

    def _save_output_hook(self, cache: ActivationCache, layer: int, hook_name: str):
        """Create a hook that saves module output to cache."""
        def hook(module: nn.Module, args, kwargs, output):
            _ = module
            _ = args
            _ = kwargs
            cache[(hook_name, layer)] = output

        return hook

    def _save_pre_hook(self, cache: ActivationCache, layer: int, hook_name: str):
        """Create a pre-hook that saves module input to cache."""
        def hook(module: nn.Module, args, kwargs):
            _ = module
            if len(args) > 0:
                cache[(hook_name, layer)] = args[0]
            elif "hidden_states" in kwargs:
                cache[(hook_name, layer)] = kwargs["hidden_states"]
            return None  # Don't modify inputs

        return hook

    def _save_input_hook_pre(self, layer: int, hook_name: str, canonical: str):
        """Create a PRE-hook that saves input data for later finalization."""
        def hook(module: nn.Module, args, kwargs):
            _ = module
            if canonical == "lm.attn.pattern":
                hook_data = {}
                if len(args) > 0:
                    hook_data["hidden_states"] = args[0]
                if len(args) > 1:
                    if isinstance(args[1], tuple):
                        hook_data["position_embeddings"] = args[1]
                    else:
                        hook_data["attention_mask"] = args[1]
                if (
                    len(args) > 2
                    and "position_ids" not in hook_data
                    and "position_embeddings" not in hook_data
                ):
                    hook_data["position_ids"] = args[2]
                elif "hidden_states" in kwargs:
                    hook_data["hidden_states"] = kwargs["hidden_states"]

                hook_data["attention_mask"] = kwargs.get(
                    "attention_mask", hook_data.get("attention_mask")
                )
                # Fix: preserve args-derived value if kwargs doesn't have it
                hook_data["position_ids"] = kwargs.get(
                    "position_ids", hook_data.get("position_ids")
                )
                hook_data["position_embeddings"] = kwargs.get(
                    "position_embeddings", hook_data.get("position_embeddings")
                )
                self._input_cache[(hook_name, layer)] = hook_data
            else:
                # For other hooks: capture first arg
                if len(args) > 0:
                    self._input_cache[(hook_name, layer)] = args[0]
                elif "hidden_states" in kwargs:
                    self._input_cache[(hook_name, layer)] = kwargs["hidden_states"]
            return None  # Don't modify inputs

        return hook

    def _save_input_hook_post(self, layer: int, hook_name: str):
        """Create a POST-hook that saves input (args) for later finalization.

        Used for lm.attn.out where we hook o_proj and capture its input.
        """
        def hook(module: nn.Module, args, kwargs, output):
            _ = module
            _ = output
            # Capture the input to the module (args[0])
            if len(args) > 0:
                self._input_cache[(hook_name, layer)] = args[0]
            elif "hidden_states" in kwargs:
                self._input_cache[(hook_name, layer)] = kwargs["hidden_states"]

        return hook

    def _wrap_pre_hook(self, hook: "Hook"):
        """Wrap a Hook object for pre-hook signature."""
        def wrapper(module, args, kwargs):
            result = hook(module, args, kwargs, None)
            if result is not None:
                return result
            return None

        return wrapper
