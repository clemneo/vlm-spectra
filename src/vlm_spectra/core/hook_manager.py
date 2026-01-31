from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hook_points import HookPoint

from vlm_spectra.models.base_adapter import ModelAdapter

if TYPE_CHECKING:
    from vlm_spectra.hooks.base import Hook


class HookManager:
    """Manages PyTorch hook lifecycle for interpretability.

    Supports TransformerLens-style hook naming:
        lm.blocks.{layer}.{hook_type}

    Examples:
        manager.register_cache_hooks(cache, ["lm.blocks.*.hook_resid_post"])
        manager.register_cache_hooks(cache, ["lm.blocks.5.attn.hook_pattern"])
    """

    def __init__(self, adapter: ModelAdapter) -> None:
        self._adapter = adapter
        self._cache_handles: List[RemovableHandle] = []
        self._patch_handles: List[RemovableHandle] = []
        self._input_cache: Dict[str, Any] = {}
        self._registered_hooks: List[str] = []

    def register_cache_hooks(self, cache: ActivationCache, names: List[str]) -> None:
        """Register hooks that capture activations into cache.

        Args:
            cache: ActivationCache to store captured activations
            names: List of hook names, may include wildcards (e.g., "lm.blocks.*.hook_resid_post")
        """
        if self._cache_handles:
            self.remove_cache_hooks()
        self._input_cache = {}
        self._registered_hooks = []

        # Expand wildcards and register hooks
        for name in names:
            if not name.startswith("lm."):
                raise ValueError("Invalid Hook Name. Only LM hooks (starting with 'lm.') are supported for now")

            expanded_names = HookPoint.expand(name, self._adapter.lm_num_layers)
            for full_name in expanded_names:
                self._register_single_cache_hook(cache, full_name)
                self._registered_hooks.append(full_name)

    def _register_single_cache_hook(self, cache: ActivationCache, name: str) -> None:
        """Register a single cache hook for a specific layer."""
        hook_type, layer = HookPoint.parse(name)
        module_getter = HookPoint.get_module_getter(hook_type)
        is_pre = HookPoint.is_pre_hook(hook_type)
        is_computed = HookPoint.is_computed(hook_type)

        # Get module from adapter
        getter_fn = getattr(self._adapter, module_getter, None)
        if getter_fn is None:
            raise NotImplementedError(
                f"Adapter does not implement {module_getter} for hook {hook_type}"
            )
        module = getter_fn(layer)

        # Computed hooks need special handling
        if is_computed:
            if hook_type == "attn.hook_pattern":
                # Capture inputs to attention for pattern computation
                hook_fn = self._save_pattern_inputs(name)
                handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
            elif hook_type == "attn.hook_head_out":
                # Capture input to o_proj for per-head decomposition
                hook_fn = self._save_o_proj_input(name)
                handle = module.register_forward_hook(hook_fn, with_kwargs=True)
            elif hook_type == "hook_resid_mid":
                raise NotImplementedError("hook_resid_mid is not yet implemented")
            elif hook_type == "attn.hook_scores":
                raise NotImplementedError("attn.hook_scores is not yet implemented")
            else:
                raise NotImplementedError(f"Computed hook {hook_type} is not yet implemented")
        elif is_pre:
            # Pre-hooks capture input to the module
            hook_fn = self._save_pre_hook(cache, name)
            handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
        else:
            # Post-hooks capture output from the module
            hook_fn = self._save_output_hook(cache, name)
            handle = module.register_forward_hook(hook_fn, with_kwargs=True)

        self._cache_handles.append(handle)

    def register_patch_hooks(self, hooks: List["Hook"]) -> None:
        """Register hooks that modify activations."""
        if self._patch_handles:
            self.remove_patch_hooks()

        for hook in hooks:
            hook_point = getattr(hook, "hook_point", "lm.blocks.0.hook_resid_post")
            layer = hook.layer

            # Handle both old and new naming conventions during transition
            if not hook_point.startswith("lm.blocks."):
                # Legacy hook point - convert to new format
                legacy_to_new = {
                    "lm.layer.pre": "hook_resid_pre",
                    "lm.layer.post": "hook_resid_post",
                    "lm.attn.out": "attn.hook_out",
                    "lm.attn.head": "attn.hook_z",
                    "lm.attn.pre": "attn.hook_in",
                    "lm.attn.pattern": "attn.hook_pattern",
                    "lm.mlp.out": "mlp.hook_out",
                }
                if hook_point in legacy_to_new:
                    hook_type = legacy_to_new[hook_point]
                    hook_point = HookPoint.format(hook_type, layer)
                else:
                    raise ValueError(f"Unknown legacy hook point: {hook_point}")

            hook_type, parsed_layer = HookPoint.parse(hook_point)
            if parsed_layer != layer and parsed_layer != "*":
                # Layer from hook_point takes precedence
                layer = parsed_layer

            # Inject num_heads for hooks that need it
            if hasattr(hook, "set_num_heads"):
                hook.set_num_heads(self._adapter.lm_num_heads)

            module_getter = HookPoint.get_module_getter(hook_type)
            getter_fn = getattr(self._adapter, module_getter)
            module = getter_fn(layer)
            is_pre = HookPoint.is_pre_hook(hook_type)

            if is_pre:
                handle = module.register_forward_pre_hook(
                    self._wrap_pre_hook(hook), with_kwargs=True
                )
            else:
                handle = module.register_forward_hook(hook, with_kwargs=True)
            self._patch_handles.append(handle)

    def finalize_cache(self, cache: ActivationCache, names: List[str]) -> None:
        """Compute derived cache entries that require extra inputs."""
        for name in self._registered_hooks:
            hook_type, layer = HookPoint.parse(name)

            if hook_type == "attn.hook_head_out":
                if name in self._input_cache:
                    concatenated_heads = self._input_cache[name]
                    cache[name] = self._adapter.compute_per_head_contributions(
                        concatenated_heads, layer
                    )

            elif hook_type == "attn.hook_pattern":
                if name in self._input_cache:
                    hook_data = self._input_cache[name]
                    if isinstance(hook_data, dict):
                        attn_patterns = self._adapter.compute_attention_patterns(
                            hidden_states=hook_data["hidden_states"],
                            layer=layer,
                            attention_mask=hook_data.get("attention_mask"),
                            position_ids=hook_data.get("position_ids"),
                            position_embeddings=hook_data.get("position_embeddings"),
                        )
                    else:
                        attn_patterns = self._adapter.compute_attention_patterns(
                            hook_data, layer
                        )
                    cache[name] = attn_patterns

    def remove_all_hooks(self) -> None:
        """Remove all registered hooks."""
        self.remove_cache_hooks()
        self.remove_patch_hooks()

    def remove_cache_hooks(self) -> None:
        """Remove cache hooks only."""
        for handle in self._cache_handles:
            handle.remove()
        self._cache_handles.clear()
        self._registered_hooks.clear()

    def remove_patch_hooks(self) -> None:
        """Remove patch hooks only."""
        for handle in self._patch_handles:
            handle.remove()
        self._patch_handles.clear()

    def _save_output_hook(self, cache: ActivationCache, name: str):
        """Create a hook that saves module output to cache."""
        def hook(module: nn.Module, args, kwargs, output):
            _ = module
            _ = args
            _ = kwargs
            cache[name] = output

        return hook

    def _save_pre_hook(self, cache: ActivationCache, name: str):
        """Create a pre-hook that saves module input to cache."""
        def hook(module: nn.Module, args, kwargs):
            _ = module
            if len(args) > 0:
                cache[name] = args[0]
            elif "hidden_states" in kwargs:
                cache[name] = kwargs["hidden_states"]
            return None  # Don't modify inputs

        return hook

    def _save_pattern_inputs(self, name: str):
        """Create a pre-hook that saves inputs for attention pattern computation."""
        def hook(module: nn.Module, args, kwargs):
            _ = module
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
            hook_data["position_ids"] = kwargs.get(
                "position_ids", hook_data.get("position_ids")
            )
            hook_data["position_embeddings"] = kwargs.get(
                "position_embeddings", hook_data.get("position_embeddings")
            )
            self._input_cache[name] = hook_data
            return None  # Don't modify inputs

        return hook

    def _save_o_proj_input(self, name: str):
        """Create a post-hook that saves input to o_proj for per-head decomposition."""
        def hook(module: nn.Module, args, kwargs, output):
            _ = module
            _ = output
            if len(args) > 0:
                self._input_cache[name] = args[0]
            elif "hidden_states" in kwargs:
                self._input_cache[name] = kwargs["hidden_states"]

        return hook

    def _wrap_pre_hook(self, hook: "Hook"):
        """Wrap a Hook object for pre-hook signature."""
        def wrapper(module, args, kwargs):
            result = hook(module, args, kwargs, None)
            if result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    new_args, new_kwargs = result
                    if isinstance(new_args, tuple) and isinstance(new_kwargs, dict):
                        return new_args, new_kwargs
                return result
            return None

        return wrapper
