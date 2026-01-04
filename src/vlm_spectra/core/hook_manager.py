from __future__ import annotations

from typing import Dict, List, Tuple

import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from vlm_spectra.core.activation_cache import ActivationCache


class HookManager:
    """Manages PyTorch hook lifecycle for interpretability."""

    _LEGACY_TO_CANONICAL = {
        "lm_resid_pre": "lm.layer.pre",
        "lm_resid_post": "lm.layer.post",
        "lm_resid_mid": "lm.layer.mid",
        "lm_attn_out": "lm.attn.out",
        "lm_attn_pattern": "lm.attn.pattern",
        "lm_mlp_out": "lm.mlp.out",
    }

    def __init__(self, adapter: "ModelAdapter") -> None:
        self._adapter = adapter
        self._handles: List[RemovableHandle] = []
        self._input_cache: Dict[Tuple[str, int], object] = {}
        self._canonical_names: Dict[str, str] = {}

    def register_cache_hooks(self, cache: ActivationCache, names: List[str]) -> None:
        """Register hooks that capture activations into cache."""
        self._input_cache = {}
        self._canonical_names = {}

        for name in names:
            if not name.startswith("lm"):
                raise NotImplementedError("Only LM hooks are supported for now")

            canonical = self._canonicalize(name)
            self._canonical_names[name] = canonical
            if canonical == "lm.layer.mid":
                raise NotImplementedError("Resid_mid hooks are not supported for now")

            for layer in range(self._adapter.lm_num_layers):
                module = self._get_module_for_hook(canonical, layer)
                if canonical in {"lm.attn.out", "lm.attn.pattern"}:
                    hook_fn = self._save_input_hook(layer, name, canonical)
                else:
                    hook_fn = self._save_output_hook(cache, layer, name)
                handle = module.register_forward_hook(hook_fn, with_kwargs=True)
                self._handles.append(handle)

    def finalize_cache(self, cache: ActivationCache, names: List[str]) -> None:
        """Compute derived cache entries that require extra inputs."""
        for name in names:
            canonical = self._canonical_names.get(name, self._canonicalize(name))
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
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_output_hook(self, cache: ActivationCache, layer: int, hook_name: str):
        def hook(module: nn.Module, args, kwargs, output):
            _ = module
            _ = args
            _ = kwargs
            cache[(hook_name, layer)] = output

        return hook

    def _save_input_hook(self, layer: int, hook_name: str, canonical: str):
        def hook(module: nn.Module, args, kwargs, output):
            _ = module
            _ = output
            if canonical == "lm.attn.pattern":
                hook_data = {}
                if len(args) > 0:
                    hook_data["hidden_states"] = args[0]
                elif "hidden_states" in kwargs:
                    hook_data["hidden_states"] = kwargs["hidden_states"]

                hook_data["attention_mask"] = kwargs.get("attention_mask")
                hook_data["position_ids"] = kwargs.get("position_ids")
                hook_data["position_embeddings"] = kwargs.get("position_embeddings")
                self._input_cache[(hook_name, layer)] = hook_data
            else:
                if len(args) > 0:
                    self._input_cache[(hook_name, layer)] = args[0]
                elif "hidden_states" in kwargs:
                    self._input_cache[(hook_name, layer)] = kwargs["hidden_states"]

        return hook

    def _get_module_for_hook(self, hook_name: str, layer: int) -> nn.Module:
        """Resolve hook name to actual module."""
        if hook_name in {"lm.layer.pre", "lm.layer.post"}:
            return self._adapter.get_lm_layer(layer)
        if hook_name == "lm.attn.out":
            return self._adapter.get_lm_o_proj(layer)
        if hook_name == "lm.mlp.out":
            return self._adapter.get_lm_mlp(layer)
        if hook_name == "lm.attn.pattern":
            return self._adapter.get_lm_attn(layer)
        raise NotImplementedError(f"Hook name {hook_name} not supported")

    def _canonicalize(self, hook_name: str) -> str:
        if hook_name in self._LEGACY_TO_CANONICAL:
            return self._LEGACY_TO_CANONICAL[hook_name]
        if "." in hook_name:
            return hook_name
        raise NotImplementedError(f"Hook name {hook_name} not supported")
