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

    def test_resid_post_matches_hidden_states(self, model):
        """Cached lm_resid_post should match output_hidden_states."""
        model.model.eval()
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)

        outputs = model.forward(inputs, output_hidden_states=True)

        tolerance = 2e-3  # bfloat16 tolerance

        # Compare non-final layers (final layer may have normalization differences)
        for layer_idx in range(model.adapter.lm_num_layers - 1):
            cache_resid = model.cache[("lm_resid_post", layer_idx)]
            # hidden_states[0] is input embeddings, so layer outputs start at index 1
            hf_hidden = outputs.hidden_states[layer_idx + 1]

            cache_on_device = cache_resid.to(hf_hidden.device)
            diff = torch.abs(cache_on_device - hf_hidden).mean()
            assert diff < tolerance, f"Layer {layer_idx}: diff={diff:.6f} > {tolerance}"

    def test_attn_pattern_matches_attentions(self, model):
        """Cached lm_attn_pattern should match output_attentions."""
        model.model.eval()
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with model.run_with_cache(["lm_attn_pattern"]):
            model.forward(inputs)

        outputs = model.forward(inputs, output_attentions=True)

        tolerance = 2e-3

        for layer_idx in range(model.adapter.lm_num_layers):
            cache_attn = model.cache[("lm_attn_pattern", layer_idx)]
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

        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)

        # HuggingFace hidden_states shape: [batch, seq_len, hidden_dim]
        sample = model.cache[("lm_resid_post", 0)]
        expected_shape = (1, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape


class TestCacheMultipleHooks:
    """Test caching multiple hook types simultaneously."""

    def test_cache_multiple_hook_types(self, model):
        """Multiple hook types should be cached together."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        hooks = ["lm_resid_post", "lm_mlp_out", "lm_attn_pattern"]
        with model.run_with_cache(hooks):
            model.forward(inputs)

        # Verify all hooks captured
        for hook_name in hooks:
            num_cached = len([k for k in model.cache.keys() if hook_name in k[0]])
            assert num_cached == model.adapter.lm_num_layers, (
                f"{hook_name}: expected {model.adapter.lm_num_layers} layers, got {num_cached}"
            )

        # Verify shapes
        resid_sample = model.cache[("lm_resid_post", 0)]
        mlp_sample = model.cache[("lm_mlp_out", 0)]
        attn_sample = model.cache[("lm_attn_pattern", 0)]

        assert resid_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert mlp_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert attn_sample.shape == (1, model.adapter.lm_num_heads, seq_len, seq_len)

    def test_canonical_and_legacy_names_equivalent(self, model):
        """Canonical and legacy hook names should produce equivalent results."""
        model.model.eval()
        image = generate_random_image(seed=123)
        inputs = model.prepare_messages("Describe.", image)

        # Cache with legacy names
        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)
        legacy_cache = model.cache[("lm_resid_post", 0)].clone()

        # Cache with canonical names
        with model.run_with_cache(["lm.layer.post"]):
            model.forward(inputs)
        canonical_cache = model.cache[("lm.layer.post", 0)].clone()

        # Should be identical
        assert torch.allclose(legacy_cache, canonical_cache, atol=1e-5)


class TestCacheUnsupportedHooks:
    """Test error handling for unsupported hooks."""

    def test_vision_hooks_raise(self, model):
        """Vision hooks should raise NotImplementedError."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with pytest.raises(NotImplementedError, match="Only LM hooks are supported"):
            with model.run_with_cache(["vision_resid_pre"]):
                model.forward(inputs)

    def test_resid_mid_raises(self, model):
        """Resid_mid hooks should raise NotImplementedError."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with pytest.raises(NotImplementedError, match="Resid_mid hooks are not supported"):
            with model.run_with_cache(["lm_resid_mid"]):
                model.forward(inputs)
