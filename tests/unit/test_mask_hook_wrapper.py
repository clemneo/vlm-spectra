"""Unit tests for _wrap_mask_patch_hook causal mask synthesis.

These tests call the wrapper directly with fake tensors — no model needed.
"""

import pytest
import torch
import torch.nn as nn

from vlm_spectra.core.hook_manager import HookManager


def _make_wrapper(hook_fn):
    """Create a mask-patch wrapper without a real HookManager."""
    # _wrap_mask_patch_hook's closure doesn't reference self, so a
    # minimal object with the method is sufficient.
    return HookManager.__dict__["_wrap_mask_patch_hook"](None, hook_fn)


def _passthrough(module, args, kwargs, mask):
    return mask


class TestCausalMaskSynthesis:
    """Tests for the SDPA None-mask path in _wrap_mask_patch_hook."""

    def test_prefill_builds_causal_mask(self):
        """When attention_mask is None during prefill, a square causal mask is built."""
        captured = {}

        def capture(module, args, kwargs, mask):
            captured["mask"] = mask
            return mask

        wrapper = _make_wrapper(capture)
        seq_len = 5
        kwargs = {
            "hidden_states": torch.zeros(1, seq_len, 8),
        }
        wrapper(nn.Identity(), (), kwargs)

        mask = captured["mask"]
        assert mask.shape == (1, 1, seq_len, seq_len)
        # Lower triangle + diagonal should be 0 (attend)
        assert (torch.tril(mask) == 0).all()
        # Upper triangle should be -inf (masked)
        upper = mask[0, 0].triu(diagonal=1)
        assert (upper[upper != 0] == float("-inf")).all()

    def test_prefill_respects_batch_size(self):
        """Synthesized mask batch dim matches hidden_states batch dim."""
        wrapper = _make_wrapper(_passthrough)
        kwargs = {
            "hidden_states": torch.zeros(3, 4, 8),
        }
        _, new_kwargs = wrapper(nn.Identity(), (), kwargs)
        assert new_kwargs["attention_mask"].shape[0] == 3

    def test_cached_decoding_raises(self):
        """Cached decoding (q_len < kv_len) should raise NotImplementedError."""
        wrapper = _make_wrapper(_passthrough)
        kwargs = {
            "hidden_states": torch.zeros(1, 1, 8),
            "cache_position": torch.tensor([5]),  # 6th token, but q_len=1
        }
        with pytest.raises(NotImplementedError, match="cached decoding"):
            wrapper(nn.Identity(), (), kwargs)

    def test_prefill_with_cache_position_allowed(self):
        """Prefill with cache_position (all positions present) should work."""
        wrapper = _make_wrapper(_passthrough)
        seq_len = 4
        kwargs = {
            "hidden_states": torch.zeros(1, seq_len, 8),
            "cache_position": torch.arange(seq_len),
        }
        # Should not raise — cache_position.max()+1 == q_len
        result = wrapper(nn.Identity(), (), kwargs)
        assert result is not None

    def test_no_hidden_states_returns_none(self):
        """If both attention_mask and hidden_states are None, return None."""
        wrapper = _make_wrapper(_passthrough)
        result = wrapper(nn.Identity(), (), {})
        assert result is None

    def test_existing_mask_unchanged(self):
        """When attention_mask is already provided, it passes through directly."""
        received = {}

        def capture(module, args, kwargs, mask):
            received["mask"] = mask
            return mask

        wrapper = _make_wrapper(capture)
        existing_mask = torch.ones(1, 1, 4, 4)
        kwargs = {"attention_mask": existing_mask}
        wrapper(nn.Identity(), (), kwargs)

        assert received["mask"] is existing_mask
