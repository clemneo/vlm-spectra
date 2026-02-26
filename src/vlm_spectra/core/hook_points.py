from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class HookConfig:
    """Configuration for a hook point."""

    module_getter: str
    is_pre: bool = False
    is_virtual: bool = False


class HookPoint:
    """Registry and parser for hook points.

    Hook names follow TransformerLens conventions:
        lm.blocks.{layer}.{hook_type}

    Examples:
        lm.blocks.5.hook_resid_post
        lm.blocks.*.attn.hook_pattern  (wildcard for all layers)
    """

    HOOK_CONFIGS: dict[str, HookConfig] = {
        # Residual stream
        "hook_resid_pre": HookConfig("get_lm_layer", is_pre=True),
        "hook_resid_post": HookConfig("get_lm_layer"),
        # Attention
        "attn.hook_in": HookConfig("get_lm_attn", is_pre=True),
        "attn.hook_q": HookConfig("get_lm_q_proj"),
        "attn.hook_k": HookConfig("get_lm_k_proj"),
        "attn.hook_v": HookConfig("get_lm_v_proj"),
        "attn.hook_scores": HookConfig("get_lm_attn", is_virtual=True),
        "attn.hook_pattern": HookConfig("get_lm_attn", is_virtual=True),
        "attn.hook_mask": HookConfig("get_lm_attn", is_pre=True),
        "attn.hook_z": HookConfig("get_lm_o_proj", is_pre=True),
        "attn.hook_out": HookConfig("get_lm_o_proj"),
        "attn.hook_head_out": HookConfig("get_lm_o_proj", is_virtual=True),
        # MLP
        "mlp.hook_in": HookConfig("get_lm_mlp", is_pre=True),
        "mlp.hook_pre": HookConfig("get_lm_gate_proj"),
        "mlp.hook_pre_linear": HookConfig("get_lm_up_proj"),
        "mlp.hook_post": HookConfig("get_lm_down_proj", is_pre=True),
        "mlp.hook_out": HookConfig("get_lm_mlp"),
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
    def get_config(cls, hook_type: str) -> HookConfig:
        """Get configuration for a hook type.

        Args:
            hook_type: Hook type like 'hook_resid_post'

        Returns:
            HookConfig with module_getter, is_pre, and is_virtual fields
        """
        if hook_type not in cls.HOOK_CONFIGS:
            raise ValueError(f"Unknown hook type: {hook_type}")
        return cls.HOOK_CONFIGS[hook_type]

    @classmethod
    def is_virtual(cls, hook_type: str) -> bool:
        """Check if hook requires computation in finalize.

        Computed hooks are not directly captured from forward passes but
        are derived from other captured data (e.g., attention patterns
        require computing from Q, K, V).
        """
        return cls.get_config(hook_type).is_virtual

    @classmethod
    def is_pre_hook(cls, hook_type: str) -> bool:
        """Check if hook should be registered as forward_pre_hook."""
        return cls.get_config(hook_type).is_pre

    @classmethod
    def get_module_getter(cls, hook_type: str) -> str:
        """Get the adapter method name for retrieving the module."""
        return cls.get_config(hook_type).module_getter

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
