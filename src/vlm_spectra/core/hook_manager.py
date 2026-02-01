from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hook_points import HookPoint
from vlm_spectra.core.patch_hooks import HookFn, validate_patch_hook_type

from vlm_spectra.models.base_adapter import ModelAdapter


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
        is_virtual = HookPoint.is_virtual(hook_type)

        # Get module from adapter
        getter_fn = getattr(self._adapter, module_getter, None)
        if getter_fn is None:
            raise NotImplementedError(
                f"Adapter does not implement {module_getter} for hook {hook_type}"
            )
        module = getter_fn(layer)

        # Computed hooks need special handling
        if is_virtual:
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
                # Reuse same capture as attn.hook_pattern
                hook_fn = self._save_pattern_inputs(name)
                handle = module.register_forward_pre_hook(hook_fn, with_kwargs=True)
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

    def register_patch_hooks(
        self,
        hooks: List[Tuple[str, Callable[[nn.Module, tuple, dict, torch.Tensor], Optional[torch.Tensor]]]],
    ) -> None:
        """Register hooks that modify activations using tuple-based API.

        Args:
            hooks: List of (hook_name, hook_fn) tuples where:
                - hook_name: Full hook name like 'lm.blocks.5.hook_resid_post'
                  or with wildcard like 'lm.blocks.*.hook_resid_post'
                - hook_fn: Callable with signature (module, args, kwargs, output) -> output | None
                  Return modified output, or None to keep original.

        Raises:
            ValueError: If hook_type is a pre-hook or virtual hook (only post-hooks
                that capture actual module outputs can be used for patching)

        Example:
            def zero_hook(module, args, kwargs, output):
                return torch.zeros_like(output)

            manager.register_patch_hooks([
                ('lm.blocks.5.hook_resid_post', zero_hook),
                ('lm.blocks.*.mlp.hook_out', ScaleActivation(0.5)),
            ])
        """
        if self._patch_handles:
            self.remove_patch_hooks()

        registrations = []

        for hook_name, hook_fn in hooks:
            expanded_names = HookPoint.expand(hook_name, self._adapter.lm_num_layers)

            for full_name in expanded_names:
                hook_type, layer = HookPoint.parse(full_name)
                validate_patch_hook_type(hook_type)

                module_getter = HookPoint.get_module_getter(hook_type)
                getter_fn = getattr(self._adapter, module_getter, None)
                if getter_fn is None:
                    raise NotImplementedError(
                        f"Adapter does not implement {module_getter} for hook {hook_type}"
                    )
                module = getter_fn(layer)

                wrapped_fn = self._wrap_patch_hook(hook_fn)
                registrations.append((module, wrapped_fn))

        # Register in reverse so prepend=True preserves caller order
        for module, wrapped_fn in reversed(registrations):
            handle = module.register_forward_hook(
                wrapped_fn,
                with_kwargs=True,
                prepend=True,
            )
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

            elif hook_type == "attn.hook_scores":
                if name in self._input_cache:
                    hook_data = self._input_cache[name]
                    if isinstance(hook_data, dict):
                        attn_scores = self._adapter.compute_attention_scores(
                            hidden_states=hook_data["hidden_states"],
                            layer=layer,
                            attention_mask=hook_data.get("attention_mask"),
                            position_ids=hook_data.get("position_ids"),
                            position_embeddings=hook_data.get("position_embeddings"),
                        )
                    else:
                        attn_scores = self._adapter.compute_attention_scores(
                            hook_data, layer
                        )
                    cache[name] = attn_scores

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
            cache[name] = self._clone_activation(output)

        return hook

    def _save_pre_hook(self, cache: ActivationCache, name: str):
        """Create a pre-hook that saves module input to cache."""
        def hook(module: nn.Module, args, kwargs):
            _ = module
            if len(args) > 0:
                cache[name] = self._clone_activation(args[0])
            elif "hidden_states" in kwargs:
                cache[name] = self._clone_activation(kwargs["hidden_states"])
            return None  # Don't modify inputs

        return hook

    def _save_pattern_inputs(self, name: str):
        """Create a pre-hook that saves inputs for attention pattern computation."""
        def hook(module: nn.Module, args, kwargs):
            _ = module
            hook_data = {}
            if len(args) > 0:
                hook_data["hidden_states"] = self._clone_activation(args[0])
            if len(args) > 1:
                if isinstance(args[1], tuple):
                    hook_data["position_embeddings"] = self._clone_activation(args[1])
                else:
                    hook_data["attention_mask"] = self._clone_activation(args[1])
            if (
                len(args) > 2
                and "position_ids" not in hook_data
                and "position_embeddings" not in hook_data
            ):
                hook_data["position_ids"] = self._clone_activation(args[2])
            elif "hidden_states" in kwargs:
                hook_data["hidden_states"] = self._clone_activation(
                    kwargs["hidden_states"]
                )

            if "attention_mask" in kwargs:
                hook_data["attention_mask"] = self._clone_activation(
                    kwargs["attention_mask"]
                )
            else:
                hook_data["attention_mask"] = hook_data.get("attention_mask")

            if "position_ids" in kwargs:
                hook_data["position_ids"] = self._clone_activation(kwargs["position_ids"])
            else:
                hook_data["position_ids"] = hook_data.get("position_ids")

            if "position_embeddings" in kwargs:
                hook_data["position_embeddings"] = self._clone_activation(
                    kwargs["position_embeddings"]
                )
            else:
                hook_data["position_embeddings"] = hook_data.get(
                    "position_embeddings"
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
                self._input_cache[name] = self._clone_activation(args[0])
            elif "hidden_states" in kwargs:
                self._input_cache[name] = self._clone_activation(kwargs["hidden_states"])

        return hook

    @staticmethod
    def _clone_activation(value):
        """Detach and clone tensors to avoid downstream in-place mutation."""

        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, tuple):
            return tuple(HookManager._clone_activation(v) for v in value)
        if isinstance(value, list):
            return [HookManager._clone_activation(v) for v in value]
        if isinstance(value, dict):
            return {k: HookManager._clone_activation(v) for k, v in value.items()}
        return value

    def _wrap_patch_hook(
        self,
        hook_fn: Callable[[nn.Module, tuple, dict, torch.Tensor], Optional[torch.Tensor]],
    ):
        """Wrap a patch hook function to handle tuple outputs from modules.

        Some modules return tuples (output, cache_state, ...). This wrapper
        extracts the first element, passes it to hook_fn, and reconstructs
        the tuple if needed.
        """
        def wrapper(module: nn.Module, args: tuple, kwargs: dict, output: Union[torch.Tensor, tuple]):
            # Handle tuple outputs (common in transformer layers)
            is_tuple = isinstance(output, tuple)
            tensor_output = output[0] if is_tuple else output

            # Call the hook function
            result = hook_fn(module, args, kwargs, tensor_output)

            if result is None:
                # Hook didn't modify anything
                return output

            # Reconstruct tuple if original was tuple
            if is_tuple:
                return (result,) + output[1:]
            return result

        return wrapper
