from __future__ import annotations

import re
from typing import Union


class HookPoint:
    """Registry and parser for hook points.

    Hook names follow TransformerLens conventions:
        lm.blocks.{layer}.{hook_type}

    Examples:
        lm.blocks.5.hook_resid_post
        lm.blocks.*.attn.hook_pattern  (wildcard for all layers)
    """

    # hook_type -> (module_getter, is_pre_hook, is_computed)
    HOOK_CONFIGS = {
        "hook_resid_pre": ("get_lm_layer", True, False),
        "hook_resid_post": ("get_lm_layer", False, False),
        "hook_resid_mid": ("get_lm_layer", False, True),  # computed
        "attn.hook_in": ("get_lm_attn", True, False),
        "attn.hook_q": ("get_lm_q_proj", False, False),
        "attn.hook_k": ("get_lm_k_proj", False, False),
        "attn.hook_v": ("get_lm_v_proj", False, False),
        "attn.hook_scores": ("get_lm_attn", False, True),  # computed
        "attn.hook_pattern": ("get_lm_attn", False, True),  # computed
        "attn.hook_z": ("get_lm_o_proj", True, False),
        "attn.hook_out": ("get_lm_o_proj", False, False),
        "attn.hook_head_out": ("get_lm_o_proj", False, True),  # computed (per-head)
        "mlp.hook_in": ("get_lm_mlp", True, False),
        "mlp.hook_pre": ("get_lm_gate_proj", False, False),
        "mlp.hook_pre_linear": ("get_lm_up_proj", False, False),
        "mlp.hook_post": ("get_lm_down_proj", True, False),
        "mlp.hook_out": ("get_lm_mlp", False, False),
    }

    # Pattern: lm.blocks.{layer}.{hook_type}
    _PATTERN = re.compile(r"^lm\.blocks\.(\d+|\*)\.(.+)$")

    @classmethod
    def parse(cls, name: str) -> tuple[str, Union[int, str]]:
        """Parse hook name into (hook_type, layer).

        Args:
            name: Full hook name like 'lm.blocks.5.hook_resid_post'

        Returns:
            Tuple of (hook_type, layer) where layer is int or '*' for wildcards

        Examples:
            >>> HookPoint.parse('lm.blocks.5.hook_resid_post')
            ('hook_resid_post', 5)
            >>> HookPoint.parse('lm.blocks.*.attn.hook_pattern')
            ('attn.hook_pattern', '*')
        """
        match = cls._PATTERN.match(name)
        if not match:
            raise ValueError(f"Invalid hook name format: {name}")

        layer_str, hook_type = match.groups()

        if hook_type not in cls.HOOK_CONFIGS:
            raise ValueError(f"Unknown hook type: {hook_type}")

        layer: Union[int, str] = "*" if layer_str == "*" else int(layer_str)
        return hook_type, layer

    @classmethod
    def expand(cls, name: str, num_layers: int) -> list[str]:
        """Expand wildcard hook name to list of concrete names.

        Args:
            name: Hook name, possibly with wildcard (e.g., 'lm.blocks.*.hook_resid_post')
            num_layers: Number of layers in the model

        Returns:
            List of expanded hook names

        Examples:
            >>> HookPoint.expand('lm.blocks.*.hook_resid_post', 3)
            ['lm.blocks.0.hook_resid_post', 'lm.blocks.1.hook_resid_post', 'lm.blocks.2.hook_resid_post']
            >>> HookPoint.expand('lm.blocks.5.hook_resid_post', 10)
            ['lm.blocks.5.hook_resid_post']
        """
        hook_type, layer = cls.parse(name)

        if layer == "*":
            return [cls.format(hook_type, i) for i in range(num_layers)]
        return [name]

    @classmethod
    def format(cls, hook_type: str, layer: int) -> str:
        """Format hook_type and layer into full hook name.

        Args:
            hook_type: Hook type like 'hook_resid_post' or 'attn.hook_pattern'
            layer: Layer index

        Returns:
            Full hook name like 'lm.blocks.5.hook_resid_post'

        Examples:
            >>> HookPoint.format('hook_resid_post', 5)
            'lm.blocks.5.hook_resid_post'
            >>> HookPoint.format('attn.hook_pattern', 0)
            'lm.blocks.0.attn.hook_pattern'
        """
        if hook_type not in cls.HOOK_CONFIGS:
            raise ValueError(f"Unknown hook type: {hook_type}")
        return f"lm.blocks.{layer}.{hook_type}"

    @classmethod
    def get_config(cls, hook_type: str) -> tuple[str, bool, bool]:
        """Get configuration for a hook type.

        Args:
            hook_type: Hook type like 'hook_resid_post'

        Returns:
            Tuple of (module_getter, is_pre_hook, is_computed)
        """
        if hook_type not in cls.HOOK_CONFIGS:
            raise ValueError(f"Unknown hook type: {hook_type}")
        return cls.HOOK_CONFIGS[hook_type]

    @classmethod
    def is_computed(cls, hook_type: str) -> bool:
        """Check if hook requires computation in finalize.

        Computed hooks are not directly captured from forward passes but
        are derived from other captured data (e.g., attention patterns
        require computing from Q, K, V).
        """
        _, _, is_computed = cls.get_config(hook_type)
        return is_computed

    @classmethod
    def is_pre_hook(cls, hook_type: str) -> bool:
        """Check if hook should be registered as forward_pre_hook."""
        _, is_pre, _ = cls.get_config(hook_type)
        return is_pre

    @classmethod
    def get_module_getter(cls, hook_type: str) -> str:
        """Get the adapter method name for retrieving the module."""
        getter, _, _ = cls.get_config(hook_type)
        return getter

    @classmethod
    def matches_pattern(cls, name: str, pattern: str) -> bool:
        """Check if a hook name matches a pattern with wildcards.

        Args:
            name: Concrete hook name like 'lm.blocks.5.hook_resid_post'
            pattern: Pattern with possible wildcard like 'lm.blocks.*.hook_resid_post'

        Returns:
            True if name matches pattern
        """
        if "*" not in pattern:
            return name == pattern

        # Convert pattern to regex
        regex_pattern = re.escape(pattern).replace(r"\*", r"\d+")
        return bool(re.fullmatch(regex_pattern, name))
