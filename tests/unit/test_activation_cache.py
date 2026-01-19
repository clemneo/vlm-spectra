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
        """Cache should support dict-like access with (name, layer) tuples."""
        cache = ActivationCache()
        tensor = torch.randn(2, 3)

        cache[("lm.attn.out", 0)] = tensor

        assert ("lm.attn.out", 0) in cache
        assert torch.equal(cache[("lm.attn.out", 0)], tensor)

    def test_cache_contains(self):
        """key in cache should return True for stored keys."""
        cache = ActivationCache()
        cache[("lm.mlp.out", 5)] = torch.randn(2, 3)

        assert ("lm.mlp.out", 5) in cache
        assert ("lm.mlp.out", 0) not in cache
        assert ("lm.attn.out", 5) not in cache

    def test_cache_get_all_layers(self):
        """get_all_layers should return dict {layer_idx: tensor} for a hook name."""
        cache = ActivationCache()
        t0 = torch.randn(2, 3)
        t1 = torch.randn(2, 3)
        t_other = torch.randn(2, 3)

        cache[("lm.mlp.out", 0)] = t0
        cache[("lm.mlp.out", 1)] = t1
        cache[("lm.attn.out", 0)] = t_other

        layers = cache.get_all_layers("lm.mlp.out")

        assert set(layers.keys()) == {0, 1}
        assert torch.equal(layers[0], t0)
        assert torch.equal(layers[1], t1)

    def test_cache_get_all_layers_empty(self):
        """get_all_layers should return empty dict for missing hook name."""
        cache = ActivationCache()
        cache[("lm.mlp.out", 0)] = torch.randn(2, 3)

        layers = cache.get_all_layers("lm.attn.out")

        assert layers == {}

    def test_cache_stack_layers(self):
        """stack_layers should return [num_layers, ...] tensor."""
        cache = ActivationCache()
        batch, seq_len, hidden_dim = 1, 10, 64

        cache[("lm.mlp.out", 0)] = torch.randn(batch, seq_len, hidden_dim)
        cache[("lm.mlp.out", 1)] = torch.randn(batch, seq_len, hidden_dim)
        cache[("lm.mlp.out", 2)] = torch.randn(batch, seq_len, hidden_dim)

        stacked = cache.stack_layers("lm.mlp.out")

        assert stacked.shape == (3, batch, seq_len, hidden_dim)

    def test_cache_stack_layers_preserves_order(self):
        """stack_layers should stack in layer index order."""
        cache = ActivationCache()

        # Add out of order
        cache[("lm.layer.post", 2)] = torch.full((2, 3), 2.0)
        cache[("lm.layer.post", 0)] = torch.full((2, 3), 0.0)
        cache[("lm.layer.post", 1)] = torch.full((2, 3), 1.0)

        stacked = cache.stack_layers("lm.layer.post")

        # Should be sorted by layer index
        assert torch.allclose(stacked[0], torch.full((2, 3), 0.0))
        assert torch.allclose(stacked[1], torch.full((2, 3), 1.0))
        assert torch.allclose(stacked[2], torch.full((2, 3), 2.0))

    def test_cache_clear(self):
        """clear() should remove all entries."""
        cache = ActivationCache()
        cache[("lm.layer.post", 0)] = torch.randn(2, 3)
        cache[("lm.layer.post", 1)] = torch.randn(2, 3)
        cache[("lm.mlp.out", 0)] = torch.randn(2, 3)

        cache.clear()

        assert cache.keys() == []
        assert ("lm.layer.post", 0) not in cache

    def test_cache_detach(self):
        """detach() should remove gradient tracking from all tensors."""
        cache = ActivationCache()
        tensor = torch.randn(2, 3, requires_grad=True)
        cache[("lm.layer.post", 0)] = tensor

        assert cache[("lm.layer.post", 0)].requires_grad is True

        cache.detach()

        assert cache[("lm.layer.post", 0)].requires_grad is False

    def test_cache_detach_multiple_tensors(self):
        """detach() should work on all cached tensors."""
        cache = ActivationCache()
        cache[("lm.layer.post", 0)] = torch.randn(2, 3, requires_grad=True)
        cache[("lm.layer.post", 1)] = torch.randn(2, 3, requires_grad=True)
        cache[("lm.mlp.out", 0)] = torch.randn(2, 3, requires_grad=True)

        cache.detach()

        for key in cache.keys():
            assert cache[key].requires_grad is False

    def test_cache_keys(self):
        """keys() should return list of (name, layer_idx) tuples."""
        cache = ActivationCache()
        cache[("lm.layer.post", 0)] = torch.randn(2, 3)
        cache[("lm.mlp.out", 5)] = torch.randn(2, 3)
        cache[("lm.attn.pattern", 3)] = torch.randn(2, 3)

        keys = cache.keys()

        assert isinstance(keys, list)
        assert len(keys) == 3
        assert ("lm.layer.post", 0) in keys
        assert ("lm.mlp.out", 5) in keys
        assert ("lm.attn.pattern", 3) in keys

    def test_cache_items(self):
        """items() should iterate over (key, tensor) pairs."""
        cache = ActivationCache()
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)
        cache[("lm.layer.post", 0)] = t1
        cache[("lm.mlp.out", 1)] = t2

        items = list(cache.items())

        assert len(items) == 2
        keys = [k for k, v in items]
        assert ("lm.layer.post", 0) in keys
        assert ("lm.mlp.out", 1) in keys

    def test_cache_overwrite(self):
        """Setting same key twice should overwrite."""
        cache = ActivationCache()
        t1 = torch.randn(2, 3)
        t2 = torch.randn(2, 3)

        cache[("lm.layer.post", 0)] = t1
        cache[("lm.layer.post", 0)] = t2

        assert torch.equal(cache[("lm.layer.post", 0)], t2)
        assert len(cache.keys()) == 1

    def test_cache_missing_key_raises(self):
        """Accessing missing key should raise KeyError."""
        cache = ActivationCache()

        with pytest.raises(KeyError):
            _ = cache[("lm.layer.post", 0)]
