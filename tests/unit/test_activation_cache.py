import torch

from vlm_spectra.core.activation_cache import ActivationCache


def test_activation_cache_basic_ops():
    cache = ActivationCache()
    tensor = torch.randn(2, 3)

    cache[("lm.attn.out", 0)] = tensor

    assert ("lm.attn.out", 0) in cache
    assert torch.equal(cache[("lm.attn.out", 0)], tensor)


def test_activation_cache_layer_access():
    cache = ActivationCache()
    cache[("lm.mlp.out", 0)] = torch.randn(2, 3)
    cache[("lm.mlp.out", 1)] = torch.randn(2, 3)
    cache[("lm.attn.out", 0)] = torch.randn(2, 3)

    layers = cache.get_all_layers("lm.mlp.out")
    assert set(layers.keys()) == {0, 1}

    stacked = cache.stack_layers("lm.mlp.out")
    assert stacked.shape[0] == 2


def test_activation_cache_clear_and_detach():
    cache = ActivationCache()
    tensor = torch.randn(2, 3, requires_grad=True)
    cache[("lm.layer.post", 0)] = tensor

    cache.detach()
    assert cache[("lm.layer.post", 0)].requires_grad is False

    cache.clear()
    assert cache.keys() == []
