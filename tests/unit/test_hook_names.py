"""Unit tests for hook name canonicalization.

These tests validate the HookPointRegistry's name mapping without loading any models.
All tests should run in < 1 second total.
"""

import pytest

from vlm_spectra.core.hook_manager import HookPointRegistry


class TestHookNameCanonicalization:
    """Test hook name normalization logic."""

    def test_legacy_to_canonical_lm_resid_post(self):
        """lm_resid_post should map to lm.layer.post."""
        result = HookPointRegistry.canonicalize("lm_resid_post")
        assert result == "lm.layer.post"

    def test_legacy_to_canonical_lm_resid_pre(self):
        """lm_resid_pre should map to lm.layer.pre."""
        result = HookPointRegistry.canonicalize("lm_resid_pre")
        assert result == "lm.layer.pre"

    def test_legacy_to_canonical_lm_attn_out(self):
        """lm_attn_out should map to lm.attn.out."""
        result = HookPointRegistry.canonicalize("lm_attn_out")
        assert result == "lm.attn.out"

    def test_legacy_to_canonical_lm_attn_head(self):
        """lm_attn_head should map to lm.attn.head."""
        result = HookPointRegistry.canonicalize("lm_attn_head")
        assert result == "lm.attn.head"

    def test_legacy_to_canonical_lm_attn_pattern(self):
        """lm_attn_pattern should map to lm.attn.pattern."""
        result = HookPointRegistry.canonicalize("lm_attn_pattern")
        assert result == "lm.attn.pattern"

    def test_legacy_to_canonical_lm_mlp_out(self):
        """lm_mlp_out should map to lm.mlp.out."""
        result = HookPointRegistry.canonicalize("lm_mlp_out")
        assert result == "lm.mlp.out"

    def test_canonical_unchanged_lm_layer_post(self):
        """lm.layer.post should stay unchanged."""
        result = HookPointRegistry.canonicalize("lm.layer.post")
        assert result == "lm.layer.post"

    def test_canonical_unchanged_lm_layer_pre(self):
        """lm.layer.pre should stay unchanged."""
        result = HookPointRegistry.canonicalize("lm.layer.pre")
        assert result == "lm.layer.pre"

    def test_canonical_unchanged_lm_attn_out(self):
        """lm.attn.out should stay unchanged."""
        result = HookPointRegistry.canonicalize("lm.attn.out")
        assert result == "lm.attn.out"

    def test_canonical_unchanged_lm_attn_pattern(self):
        """lm.attn.pattern should stay unchanged."""
        result = HookPointRegistry.canonicalize("lm.attn.pattern")
        assert result == "lm.attn.pattern"

    def test_canonical_unchanged_lm_mlp_out(self):
        """lm.mlp.out should stay unchanged."""
        result = HookPointRegistry.canonicalize("lm.mlp.out")
        assert result == "lm.mlp.out"

    def test_invalid_hook_name_raises(self):
        """Unknown hook name should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Unknown hook point"):
            HookPointRegistry.canonicalize("invalid_hook")

    def test_invalid_hook_name_typo_raises(self):
        """Hook name typo should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Unknown hook point"):
            HookPointRegistry.canonicalize("lm_resid_pots")  # typo

    def test_invalid_vision_hook_raises(self):
        """Vision hook names should raise (not implemented)."""
        with pytest.raises(NotImplementedError, match="Unknown hook point"):
            HookPointRegistry.canonicalize("vision_resid_post")


class TestHookPointIsPreHook:
    """Test is_pre_hook detection."""

    def test_lm_layer_pre_is_pre_hook(self):
        """lm.layer.pre should be a pre-hook."""
        assert HookPointRegistry.is_pre_hook("lm.layer.pre") is True
        assert HookPointRegistry.is_pre_hook("lm_resid_pre") is True

    def test_lm_layer_post_is_post_hook(self):
        """lm.layer.post should be a post-hook."""
        assert HookPointRegistry.is_pre_hook("lm.layer.post") is False
        assert HookPointRegistry.is_pre_hook("lm_resid_post") is False

    def test_lm_mlp_out_is_post_hook(self):
        """lm.mlp.out should be a post-hook."""
        assert HookPointRegistry.is_pre_hook("lm.mlp.out") is False
        assert HookPointRegistry.is_pre_hook("lm_mlp_out") is False

    def test_lm_attn_pattern_is_pre_hook(self):
        """lm.attn.pattern should be a pre-hook (captures inputs to attn)."""
        assert HookPointRegistry.is_pre_hook("lm.attn.pattern") is True
        assert HookPointRegistry.is_pre_hook("lm_attn_pattern") is True


class TestHookPointRegistry:
    """Test HookPointRegistry structure."""

    def test_all_legacy_names_have_mappings(self):
        """All legacy names in LEGACY_TO_CANONICAL should map to valid hook points."""
        for legacy_name, canonical in HookPointRegistry.LEGACY_TO_CANONICAL.items():
            # Skip lm.layer.mid which is special-cased
            if canonical == "lm.layer.mid":
                continue
            assert canonical in HookPointRegistry.HOOK_POINTS, (
                f"Legacy {legacy_name} maps to {canonical} but that's not in HOOK_POINTS"
            )

    def test_hook_points_have_required_keys(self):
        """Each hook point config should have module_getter and is_pre_hook."""
        for hook_name, config in HookPointRegistry.HOOK_POINTS.items():
            assert "module_getter" in config, f"{hook_name} missing module_getter"
            assert "is_pre_hook" in config, f"{hook_name} missing is_pre_hook"
