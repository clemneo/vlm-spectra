"""Cache correctness tests.

These tests verify that cached activations match HuggingFace's
output_hidden_states and output_attentions within tolerance.
"""

import torch
import pytest
from PIL import Image
import numpy as np


def generate_random_image(width=56, height=56, seed=None):
    """Generate a random RGB image."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestCacheCorrectness:
    """Verify cached activations match HuggingFace outputs."""

    @pytest.mark.requires_capability("strict_residual_stream")
    def test_resid_post_matches_hidden_states(self, model):
        """Cached hook_resid_post should match output_hidden_states."""
        model.model.eval()
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            model.forward(inputs)

        outputs = model.forward(inputs, output_hidden_states=True)

        tolerance = 2e-3  # bfloat16 tolerance

        # Compare non-final layers (final layer may have normalization differences)
        for layer_idx in range(model.adapter.lm_num_layers - 1):
            cache_resid = model.cache[f"lm.blocks.{layer_idx}.hook_resid_post"]
            # hidden_states[0] is input embeddings, so layer outputs start at index 1
            hf_hidden = outputs.hidden_states[layer_idx + 1]

            cache_on_device = cache_resid.to(hf_hidden.device)
            diff = torch.abs(cache_on_device - hf_hidden).mean()
            assert diff < tolerance, f"Layer {layer_idx}: diff={diff:.6f} > {tolerance}"

    def test_attn_pattern_matches_attentions(self, model):
        """Cached attn.hook_pattern should match output_attentions."""
        model.model.eval()
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with model.run_with_cache(["lm.blocks.*.attn.hook_pattern"]):
            model.forward(inputs)

        outputs = model.forward(inputs, output_attentions=True)

        tolerance = 2e-3

        for layer_idx in range(model.adapter.lm_num_layers):
            cache_attn = model.cache[f"lm.blocks.{layer_idx}.attn.hook_pattern"]
            hf_attn = outputs.attentions[layer_idx]

            assert cache_attn.shape == hf_attn.shape, (
                f"Layer {layer_idx}: shape mismatch {cache_attn.shape} vs {hf_attn.shape}"
            )
            cache_on_device = cache_attn.to(hf_attn.device)
            diff = torch.abs(cache_on_device - hf_attn).mean()
            assert diff < tolerance, f"Layer {layer_idx}: diff={diff:.6f} > {tolerance}"

    def test_cache_shapes_match_hf_shapes(self, model):
        """Cached activation shapes should match HuggingFace shapes."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            model.forward(inputs)

        # HuggingFace hidden_states shape: [batch, seq_len, hidden_dim]
        sample = model.cache["lm.blocks.0.hook_resid_post"]
        expected_shape = (1, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape


class TestCacheMultipleHooks:
    """Test caching multiple hook types simultaneously."""

    def test_cache_multiple_hook_types(self, model):
        """Multiple hook types should be cached together."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        hooks = [
            "lm.blocks.*.hook_resid_post",
            "lm.blocks.*.mlp.hook_out",
            "lm.blocks.*.attn.hook_pattern",
        ]
        with model.run_with_cache(hooks):
            model.forward(inputs)

        # Verify all hooks captured
        resid_count = len([k for k in model.cache.keys() if "hook_resid_post" in k])
        mlp_count = len([k for k in model.cache.keys() if "mlp.hook_out" in k])
        attn_count = len([k for k in model.cache.keys() if "attn.hook_pattern" in k])

        assert resid_count == model.adapter.lm_num_layers
        assert mlp_count == model.adapter.lm_num_layers
        assert attn_count == model.adapter.lm_num_layers

        # Verify shapes
        resid_sample = model.cache["lm.blocks.0.hook_resid_post"]
        mlp_sample = model.cache["lm.blocks.0.mlp.hook_out"]
        attn_sample = model.cache["lm.blocks.0.attn.hook_pattern"]

        assert resid_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert mlp_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert attn_sample.shape == (1, model.adapter.lm_num_heads, seq_len, seq_len)


class TestCacheUnsupportedHooks:
    """Test error handling for unsupported hooks."""
    def test_invalid_hook_format_raises(self, model):
        """Invalid hook format should raise ValueError."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with pytest.raises(ValueError):
            with model.run_with_cache(["lm_resid_post"]):  # old format
                model.forward(inputs)
