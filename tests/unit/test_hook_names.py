"""Unit tests for HookPoint parsing and formatting.

These tests validate the HookPoint class without loading any models.
All tests should run in < 1 second total.
"""

import pytest

from vlm_spectra.core.hook_points import HookPoint
from vlm_spectra.core.patch_hooks import validate_patch_hook_type


class TestHookPointParse:
    """Test HookPoint.parse() method."""

    def test_parse_resid_post(self):
        """Parse residual post hook."""
        hook_type, layer = HookPoint.parse("lm.blocks.5.hook_resid_post")
        assert hook_type == "hook_resid_post"
        assert layer == 5

    def test_parse_attn_pattern(self):
        """Parse attention pattern hook."""
        hook_type, layer = HookPoint.parse("lm.blocks.12.attn.hook_pattern")
        assert hook_type == "attn.hook_pattern"
        assert layer == 12

    def test_parse_mlp_hook(self):
        """Parse MLP output hook."""
        hook_type, layer = HookPoint.parse("lm.blocks.0.mlp.hook_out")
        assert hook_type == "mlp.hook_out"
        assert layer == 0

    def test_parse_wildcard(self):
        """Parse wildcard layer."""
        hook_type, layer = HookPoint.parse("lm.blocks.*.hook_resid_post")
        assert hook_type == "hook_resid_post"
        assert layer == "*"

    def test_parse_invalid_format_raises(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hook name format"):
            HookPoint.parse("lm_resid_post")  # old format

    def test_parse_unknown_hook_type_raises(self):
        """Unknown hook type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown hook type"):
            HookPoint.parse("lm.blocks.5.hook_unknown")


class TestHookPointExpand:
    """Test HookPoint.expand() method."""

    def test_expand_wildcard(self):
        """Expand wildcard to all layers."""
        names = HookPoint.expand("lm.blocks.*.hook_resid_post", num_layers=3)
        assert names == [
            "lm.blocks.0.hook_resid_post",
            "lm.blocks.1.hook_resid_post",
            "lm.blocks.2.hook_resid_post",
        ]

    def test_expand_specific_layer(self):
        """Specific layer returns single-item list."""
        names = HookPoint.expand("lm.blocks.5.hook_resid_post", num_layers=10)
        assert names == ["lm.blocks.5.hook_resid_post"]


class TestHookPointFormat:
    """Test HookPoint.format() method."""

    def test_format_resid_post(self):
        """Format residual post hook."""
        name = HookPoint.format("hook_resid_post", 5)
        assert name == "lm.blocks.5.hook_resid_post"

    def test_format_attn_pattern(self):
        """Format attention pattern hook."""
        name = HookPoint.format("attn.hook_pattern", 12)
        assert name == "lm.blocks.12.attn.hook_pattern"

    def test_format_unknown_raises(self):
        """Unknown hook type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown hook type"):
            HookPoint.format("hook_unknown", 0)


class TestHookPointConfig:
    """Test HookPoint configuration methods."""

    def test_is_pre_hook_resid_pre(self):
        """hook_resid_pre should be a pre-hook."""
        assert HookPoint.is_pre_hook("hook_resid_pre") is True

    def test_is_pre_hook_resid_post(self):
        """hook_resid_post should be a post-hook."""
        assert HookPoint.is_pre_hook("hook_resid_post") is False

    def test_is_virtual_pattern(self):
        """attn.hook_pattern should be computed."""
        assert HookPoint.is_virtual("attn.hook_pattern") is True

    def test_is_virtual_resid_post(self):
        """hook_resid_post should not be computed."""
        assert HookPoint.is_virtual("hook_resid_post") is False

    def test_all_hook_types_have_config(self):
        """All hook types in HOOK_CONFIGS should have valid config."""
        for hook_type in HookPoint.HOOK_CONFIGS:
            config = HookPoint.get_config(hook_type)
            assert isinstance(config.module_getter, str)
            assert isinstance(config.is_pre, bool)
            assert isinstance(config.is_virtual, bool)


class TestAttnHookMask:
    """Tests for the attn.hook_mask hook point."""

    def test_parse_mask_hook(self):
        hook_type, layer = HookPoint.parse("lm.blocks.3.attn.hook_mask")
        assert hook_type == "attn.hook_mask"
        assert layer == 3

    def test_format_mask_hook(self):
        name = HookPoint.format("attn.hook_mask", 7)
        assert name == "lm.blocks.7.attn.hook_mask"

    def test_expand_wildcard_mask_hook(self):
        names = HookPoint.expand("lm.blocks.*.attn.hook_mask", num_layers=4)
        assert names == [
            "lm.blocks.0.attn.hook_mask",
            "lm.blocks.1.attn.hook_mask",
            "lm.blocks.2.attn.hook_mask",
            "lm.blocks.3.attn.hook_mask",
        ]

    def test_is_pre_hook(self):
        assert HookPoint.is_pre_hook("attn.hook_mask") is True

    def test_is_not_virtual(self):
        assert HookPoint.is_virtual("attn.hook_mask") is False

    def test_module_getter(self):
        assert HookPoint.get_module_getter("attn.hook_mask") == "get_lm_attn"

    def test_validate_patch_hook_type_returns_true(self):
        """attn.hook_mask should be accepted as a valid pre-hook for patching."""
        assert validate_patch_hook_type("attn.hook_mask") is True
