"""Unit tests for ActivationCache API.

These tests validate the ActivationCache class in isolation without loading any models.
All tests should run in < 1 second total.
"""

import torch
import pytest

from vlm_spectra.core.activation_cache import ActivationCache


class TestActivationCacheAPI:
    """Test the ActivationCache API directly without model loading."""

    def test_cache_dict_access(self):
        """Cache should support dict-like access with string keys."""
        cache = ActivationCache()
        tensor = torch.randn(2, 3)

        cache["lm.blocks.0.attn.hook_out"] = tensor

        assert "lm.blocks.0.attn.hook_out" in cache
        assert torch.equal(cache["lm.blocks.0.attn.hook_out"], tensor)

    def test_cache_contains(self):
        """key in cache should return True for stored keys."""
        cache = ActivationCache()
        cache["lm.blocks.5.mlp.hook_out"] = torch.randn(2, 3)

        assert "lm.blocks.5.mlp.hook_out" in cache
        assert "lm.blocks.0.mlp.hook_out" not in cache
        assert "lm.blocks.5.attn.hook_out" not in cache

    def test_cache_stack_preserves_order(self):
        """stack() should stack in layer index order."""
        cache = ActivationCache()

        # Add out of order
        cache["lm.blocks.2.hook_resid_post"] = torch.full((2, 3), 2.0)
        cache["lm.blocks.0.hook_resid_post"] = torch.full((2, 3), 0.0)
        cache["lm.blocks.1.hook_resid_post"] = torch.full((2, 3), 1.0)

        stacked = cache.stack("lm.blocks.*.hook_resid_post")

        assert stacked.shape == (3, 2, 3)
        assert torch.allclose(stacked[0], torch.full((2, 3), 0.0))
        assert torch.allclose(stacked[1], torch.full((2, 3), 1.0))
        assert torch.allclose(stacked[2], torch.full((2, 3), 2.0))

    def test_cache_clear(self):
        """clear() should remove all entries."""
        cache = ActivationCache()
        cache["lm.blocks.0.hook_resid_post"] = torch.randn(2, 3)
        cache["lm.blocks.1.hook_resid_post"] = torch.randn(2, 3)

        cache.clear()

        assert len(cache) == 0
        assert "lm.blocks.0.hook_resid_post" not in cache

    def test_cache_detach(self):
        """detach() should remove gradient tracking from all tensors."""
        cache = ActivationCache()
        cache["lm.blocks.0.hook_resid_post"] = torch.randn(2, 3, requires_grad=True)
        cache["lm.blocks.1.hook_resid_post"] = torch.randn(2, 3, requires_grad=True)

        cache.detach()

        for key in cache:
            assert cache[key].requires_grad is False

    def test_cache_missing_key_raises(self):
        """Accessing missing key should raise KeyError."""
        cache = ActivationCache()

        with pytest.raises(KeyError):
            _ = cache["lm.blocks.0.hook_resid_post"]
