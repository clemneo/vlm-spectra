"""Integration tests for hook registration mechanics.

These tests verify that hooks can be registered, fire correctly,
and are properly cleaned up using SmolVLM-256M.
"""

import pytest
import torch
from PIL import Image
import numpy as np


def generate_test_image(width=56, height=56, seed=None):
    """Generate a simple test image."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestHookRegistration:
    """Test hook registration on real modules."""

    def test_hook_registers_on_module(self, tiny_model):
        """Hook should register and return handle."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm_resid_post"]):
            # Should not raise
            tiny_model.forward(inputs)

        # Cache should have data
        assert tiny_model.cache is not None
        assert len(tiny_model.cache.keys()) > 0

    def test_hook_fires_on_forward(self, tiny_model):
        """Hook callback should be invoked during forward pass."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm_resid_post"]):
            tiny_model.forward(inputs)

        # Should have captured activations for all layers
        num_cached = len([k for k in tiny_model.cache.keys() if "lm_resid_post" in k[0]])
        assert num_cached == tiny_model.adapter.lm_num_layers

    def test_hook_removed_after_context(self, tiny_model):
        """Hooks should be removed after context manager exits."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Get baseline output
        baseline = tiny_model.forward(inputs).logits.clone()

        # Run with cache context
        with tiny_model.run_with_cache(["lm_resid_post"]):
            pass  # Just enter and exit

        # After context, forward should work normally
        after = tiny_model.forward(inputs).logits

        # Outputs should be identical (no hooks interfering)
        assert torch.allclose(baseline, after, atol=1e-5)

    def test_multiple_hooks_same_module(self, tiny_model):
        """Multiple hooks on same module should coexist."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Register multiple hook types
        hooks = ["lm_resid_post", "lm_mlp_out"]
        with tiny_model.run_with_cache(hooks):
            tiny_model.forward(inputs)

        # Both should be captured
        resid_count = len([k for k in tiny_model.cache.keys() if "lm_resid_post" in k[0]])
        mlp_count = len([k for k in tiny_model.cache.keys() if "lm_mlp_out" in k[0]])

        assert resid_count == tiny_model.adapter.lm_num_layers
        assert mlp_count == tiny_model.adapter.lm_num_layers


class TestPatchHookRegistration:
    """Test patch hook registration."""

    def test_patch_hook_modifies_output(self, tiny_model):
        """Patch hook should modify layer output."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Get original output
        original = tiny_model.forward(inputs).logits.clone()

        # Define a zeroing hook
        class ZeroingHook:
            hook_point = "lm.layer.post"

            def __init__(self, layer):
                self.layer = layer

            def __call__(self, module, args, kwargs, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)

        hook = ZeroingHook(layer=0)
        with tiny_model.run_with_hooks([hook]):
            modified = tiny_model.forward(inputs).logits

        # Output should be different
        assert not torch.allclose(original, modified, atol=1e-3)

    def test_patch_hook_removed_after_context(self, tiny_model):
        """Patch hooks should be removed after context exits."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Get baseline
        baseline = tiny_model.forward(inputs).logits.clone()

        class ZeroingHook:
            hook_point = "lm.layer.post"

            def __init__(self, layer):
                self.layer = layer

            def __call__(self, module, args, kwargs, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)

        hook = ZeroingHook(layer=0)
        with tiny_model.run_with_hooks([hook]):
            pass  # Just enter and exit

        # After context, output should match baseline
        after = tiny_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)

    def test_cache_and_patch_hooks_nest(self, tiny_model):
        """Cache and patch hooks should work together."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        class ZeroingHook:
            hook_point = "lm.layer.post"

            def __init__(self, layer):
                self.layer = layer

            def __call__(self, module, args, kwargs, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)

        hook = ZeroingHook(layer=0)
        with tiny_model.run_with_cache(["lm_resid_post"]):
            with tiny_model.run_with_hooks([hook]):
                tiny_model.forward(inputs)

        # Cache should still work
        assert tiny_model.cache is not None
        assert ("lm_resid_post", 0) in tiny_model.cache


class TestHookShapes:
    """Test that hooks capture correct shapes."""

    def test_resid_post_shape(self, tiny_model):
        """lm_resid_post should have [batch, seq_len, hidden_dim] shape."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)
        seq_len = inputs["input_ids"].shape[1]

        with tiny_model.run_with_cache(["lm_resid_post"]):
            tiny_model.forward(inputs)

        sample = tiny_model.cache[("lm_resid_post", 0)]
        expected_shape = (1, seq_len, tiny_model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_mlp_out_shape(self, tiny_model):
        """lm_mlp_out should have [batch, seq_len, hidden_dim] shape."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)
        seq_len = inputs["input_ids"].shape[1]

        with tiny_model.run_with_cache(["lm_mlp_out"]):
            tiny_model.forward(inputs)

        sample = tiny_model.cache[("lm_mlp_out", 0)]
        expected_shape = (1, seq_len, tiny_model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_all_layers_same_shape(self, tiny_model):
        """All layers should have consistent hidden state shapes."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm_resid_post"]):
            tiny_model.forward(inputs)

        shapes = [
            tiny_model.cache[("lm_resid_post", i)].shape
            for i in range(tiny_model.adapter.lm_num_layers)
        ]
        assert all(s == shapes[0] for s in shapes)
