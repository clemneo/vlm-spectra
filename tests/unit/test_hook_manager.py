import torch
import torch.nn as nn

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hook_manager import HookManager


class DummyLayer(nn.Module):
    def forward(self, hidden_states, **kwargs):
        _ = kwargs
        return hidden_states + 1


class DummyAttn(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        **kwargs,
    ):
        _ = attention_mask
        _ = position_ids
        _ = position_embeddings
        _ = kwargs
        return hidden_states


class DummyAdapter:
    def __init__(self, num_layers=2, num_heads=2, hidden_dim=4):
        self.lm_num_layers = num_layers
        self.lm_num_heads = num_heads
        self.lm_hidden_dim = hidden_dim
        self.lm_head_dim = hidden_dim // num_heads
        self._layers = nn.ModuleList([DummyLayer() for _ in range(num_layers)])
        self._attn = nn.ModuleList([DummyAttn() for _ in range(num_layers)])
        self._mlp = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        self._o_proj = nn.ModuleList([nn.Identity() for _ in range(num_layers)])

    def get_lm_layer(self, layer_idx):
        return self._layers[layer_idx]

    def get_lm_attn(self, layer_idx):
        return self._attn[layer_idx]

    def get_lm_o_proj(self, layer_idx):
        return self._o_proj[layer_idx]

    def get_lm_mlp(self, layer_idx):
        return self._mlp[layer_idx]

    def compute_per_head_contributions(self, concatenated_heads, layer):
        _ = layer
        batch, seq, _ = concatenated_heads.shape
        return torch.zeros(
            batch, seq, self.lm_num_heads, self.lm_hidden_dim, device=concatenated_heads.device
        )

    def compute_attention_patterns(
        self, hidden_states, layer, attention_mask=None, position_ids=None, position_embeddings=None
    ):
        _ = layer
        _ = attention_mask
        _ = position_ids
        _ = position_embeddings
        batch, seq, _ = hidden_states.shape
        return torch.zeros(
            batch, self.lm_num_heads, seq, seq, device=hidden_states.device
        )


def test_hook_manager_register_finalize():
    adapter = DummyAdapter()
    hook_manager = HookManager(adapter)
    cache = ActivationCache()
    names = ["lm_resid_post", "lm_mlp_out", "lm_attn_out", "lm_attn_pattern"]

    hook_manager.register_cache_hooks(cache, names)

    hidden = torch.randn(1, 3, adapter.lm_hidden_dim)
    for layer in range(adapter.lm_num_layers):
        adapter.get_lm_layer(layer)(hidden)
        adapter.get_lm_mlp(layer)(hidden)
        adapter.get_lm_o_proj(layer)(hidden)
        adapter.get_lm_attn(layer)(hidden, attention_mask=torch.zeros(1, 1, 3, 3))

    hook_manager.remove_all_hooks()
    hook_manager.finalize_cache(cache, names)

    for name in names:
        for layer in range(adapter.lm_num_layers):
            assert (name, layer) in cache

    attn_out = cache[("lm_attn_out", 0)]
    assert attn_out.shape == (1, 3, adapter.lm_num_heads, adapter.lm_hidden_dim)

    attn_pattern = cache[("lm_attn_pattern", 0)]
    assert attn_pattern.shape == (1, adapter.lm_num_heads, 3, 3)


def test_hook_manager_remove_hooks_clears_handles():
    adapter = DummyAdapter()
    hook_manager = HookManager(adapter)
    cache = ActivationCache()

    hook_manager.register_cache_hooks(cache, ["lm_resid_post"])
    assert hook_manager._handles

    hook_manager.remove_all_hooks()
    assert hook_manager._handles == []
