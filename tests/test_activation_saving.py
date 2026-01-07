"""Tests for forward hooks that save activations - Concern #3.

These tests validate that a model implementation correctly:
- Captures activations at various hook points
- Produces activations with correct shapes
- Matches HuggingFace's output_hidden_states and output_attentions
"""

import torch
import pytest

from vlm_spectra.core.activation_cache import ActivationCache
from conftest import generate_random_image, generate_checkered_image


class TestRunWithCache:
    """Test the run_with_cache context manager."""

    def test_cache_lm_resid_pre(self, model):
        """Cache should capture lm_resid_pre (pre-layer residual)."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_resid_pre"]):
            model.forward(inputs)

        assert model.cache is not None
        # Check all layers are present
        num_cached = len([k for k in model.cache.keys() if "lm_resid_pre" in k[0]])
        assert num_cached == model.adapter.lm_num_layers

        # Check shape
        sample = model.cache[("lm_resid_pre", 0)]
        expected_shape = (1, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_cache_lm_resid_post(self, model):
        """Cache should capture lm_resid_post (post-layer residual)."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)

        assert model.cache is not None
        num_cached = len([k for k in model.cache.keys() if "lm_resid_post" in k[0]])
        assert num_cached == model.adapter.lm_num_layers

        sample = model.cache[("lm_resid_post", 0)]
        expected_shape = (1, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_cache_lm_attn_out(self, model):
        """Cache should capture lm_attn_out (per-head attention outputs)."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_attn_out"]):
            model.forward(inputs)

        assert model.cache is not None
        num_cached = len([k for k in model.cache.keys() if "lm_attn_out" in k[0]])
        assert num_cached == model.adapter.lm_num_layers

        sample = model.cache[("lm_attn_out", 0)]
        # Shape: (batch, seq_len, num_heads, hidden_dim)
        expected_shape = (1, seq_len, model.adapter.lm_num_heads, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_cache_lm_mlp_out(self, model):
        """Cache should capture lm_mlp_out (MLP outputs)."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_mlp_out"]):
            model.forward(inputs)

        assert model.cache is not None
        num_cached = len([k for k in model.cache.keys() if "lm_mlp_out" in k[0]])
        assert num_cached == model.adapter.lm_num_layers

        sample = model.cache[("lm_mlp_out", 0)]
        expected_shape = (1, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_cache_lm_attn_pattern(self, model):
        """Cache should capture lm_attn_pattern (attention weights)."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_attn_pattern"]):
            model.forward(inputs)

        assert model.cache is not None
        num_cached = len([k for k in model.cache.keys() if "lm_attn_pattern" in k[0]])
        assert num_cached == model.adapter.lm_num_layers

        sample = model.cache[("lm_attn_pattern", 0)]
        # Shape: (batch, num_heads, seq_len, seq_len)
        expected_shape = (1, model.adapter.lm_num_heads, seq_len, seq_len)
        assert sample.shape == expected_shape

    def test_cache_all_layers_present(self, model):
        """Cache should have entries for all layers."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)

        for layer_idx in range(model.adapter.lm_num_layers):
            assert ("lm_resid_post", layer_idx) in model.cache

    def test_cache_shapes_consistent(self, model):
        """All layers should have consistent shapes."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)

        shapes = [model.cache[("lm_resid_post", i)].shape for i in range(model.adapter.lm_num_layers)]
        # All shapes should be identical
        assert all(s == shapes[0] for s in shapes)

    def test_cache_matches_output_hidden_states(self, model):
        """lm_resid_post should match output_hidden_states for non-final layers."""
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

    def test_cache_matches_output_attentions(self, model):
        """lm_attn_pattern should match output_attentions."""
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

            assert cache_attn.shape == hf_attn.shape
            cache_on_device = cache_attn.to(hf_attn.device)
            diff = torch.abs(cache_on_device - hf_attn).mean()
            assert diff < tolerance, f"Layer {layer_idx}: diff={diff:.6f} > {tolerance}"

    def test_unsupported_hooks_raise(self, model):
        """Unsupported hook positions should raise errors."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        with pytest.raises(NotImplementedError, match="Only LM hooks are supported"):
            with model.run_with_cache(["vision_resid_pre"]):
                model.forward(inputs)

        with pytest.raises(NotImplementedError, match="Resid_mid hooks are not supported"):
            with model.run_with_cache(["lm_resid_mid"]):
                model.forward(inputs)

    def test_cache_multiple_hook_names(self, model):
        """Cache should support multiple hook names in one run."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        hooks = ["lm_resid_post", "lm_mlp_out", "lm_attn_pattern"]
        with model.run_with_cache(hooks):
            model.forward(inputs)

        for hook_name in hooks:
            num_cached = len([k for k in model.cache.keys() if hook_name in k[0]])
            assert num_cached == model.adapter.lm_num_layers

        resid_sample = model.cache[("lm_resid_post", 0)]
        mlp_sample = model.cache[("lm_mlp_out", 0)]
        attn_sample = model.cache[("lm_attn_pattern", 0)]
        assert resid_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert mlp_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert attn_sample.shape == (1, model.adapter.lm_num_heads, seq_len, seq_len)

    def test_cache_accepts_canonical_hook_names(self, model):
        """Cache should support canonical hook names."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        seq_len = inputs["input_ids"].shape[1]

        hooks = ["lm.layer.post", "lm.mlp.out", "lm.attn.pattern"]
        with model.run_with_cache(hooks):
            model.forward(inputs)

        for hook_name in hooks:
            num_cached = len([k for k in model.cache.keys() if hook_name in k[0]])
            assert num_cached == model.adapter.lm_num_layers

        resid_sample = model.cache[("lm.layer.post", 0)]
        mlp_sample = model.cache[("lm.mlp.out", 0)]
        attn_sample = model.cache[("lm.attn.pattern", 0)]
        assert resid_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert mlp_sample.shape == (1, seq_len, model.adapter.lm_hidden_dim)
        assert attn_sample.shape == (1, model.adapter.lm_num_heads, seq_len, seq_len)


class TestActivationCacheAPI:
    """Test the ActivationCache API directly."""

    def test_cache_dict_access(self):
        """Cache should support dict-like access."""
        cache = ActivationCache()
        tensor = torch.randn(2, 3)

        cache[("lm.attn.out", 0)] = tensor

        assert ("lm.attn.out", 0) in cache
        assert torch.equal(cache[("lm.attn.out", 0)], tensor)

    def test_cache_get_all_layers(self):
        """get_all_layers should return all layers for a hook name."""
        cache = ActivationCache()
        cache[("lm.mlp.out", 0)] = torch.randn(2, 3)
        cache[("lm.mlp.out", 1)] = torch.randn(2, 3)
        cache[("lm.attn.out", 0)] = torch.randn(2, 3)

        layers = cache.get_all_layers("lm.mlp.out")
        assert set(layers.keys()) == {0, 1}

    def test_cache_stack_layers(self):
        """stack_layers should stack all layers for a hook name."""
        cache = ActivationCache()
        cache[("lm.mlp.out", 0)] = torch.randn(2, 3)
        cache[("lm.mlp.out", 1)] = torch.randn(2, 3)

        stacked = cache.stack_layers("lm.mlp.out")
        assert stacked.shape[0] == 2  # 2 layers

    def test_cache_clear(self):
        """clear should remove all entries."""
        cache = ActivationCache()
        cache[("lm.layer.post", 0)] = torch.randn(2, 3)

        cache.clear()
        assert cache.keys() == []

    def test_cache_detach(self):
        """detach should remove gradient tracking."""
        cache = ActivationCache()
        tensor = torch.randn(2, 3, requires_grad=True)
        cache[("lm.layer.post", 0)] = tensor

        cache.detach()
        assert cache[("lm.layer.post", 0)].requires_grad is False


class TestRunWithCacheBatch:
    """Test run_with_cache with batched inputs."""

    def test_batch_cache_shapes_lm_resid_post(self, model):
        """Batched cache should include batch dimension and correct shapes."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [
            generate_checkered_image(width=56, height=56, seed=1),
            generate_checkered_image(width=56, height=56, seed=2),
        ]
        inputs = model.prepare_messages_batch(tasks, images)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_resid_post"]):
            model.forward_batch(tasks=tasks, images=images)

        num_cached = len([k for k in model.cache.keys() if "lm_resid_post" in k[0]])
        assert num_cached == model.adapter.lm_num_layers

        sample = model.cache[("lm_resid_post", 0)]
        expected_shape = (2, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape

    def test_batch_cache_matches_output_hidden_states(self, model):
        """Batched lm_resid_post should match output_hidden_states."""
        model.model.eval()
        tasks = ["Describe the image.", "What do you see?"]
        images = [
            generate_checkered_image(width=56, height=56, seed=3),
            generate_checkered_image(width=56, height=56, seed=4),
        ]

        with model.run_with_cache(["lm_resid_post"]):
            model.forward_batch(tasks=tasks, images=images)

        outputs = model.forward_batch(tasks=tasks, images=images, output_hidden_states=True)

        tolerance = 2e-3
        for layer_idx in range(model.adapter.lm_num_layers - 1):
            cache_resid = model.cache[("lm_resid_post", layer_idx)]
            hf_hidden = outputs.hidden_states[layer_idx + 1]

            cache_on_device = cache_resid.to(hf_hidden.device)
            diff = torch.abs(cache_on_device - hf_hidden).mean()
            assert diff < tolerance, f"Layer {layer_idx}: diff={diff:.6f} > {tolerance}"

    def test_batch_cache_attn_pattern_shape(self, model):
        """Batched lm_attn_pattern should include batch dimension."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [
            generate_random_image(width=56, height=56, seed=5),
            generate_random_image(width=56, height=56, seed=6),
        ]
        inputs = model.prepare_messages_batch(tasks, images)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm_attn_pattern"]):
            model.forward_batch(tasks=tasks, images=images)

        sample = model.cache[("lm_attn_pattern", 0)]
        expected_shape = (2, model.adapter.lm_num_heads, seq_len, seq_len)
        assert sample.shape == expected_shape

    def test_batch_single_item_matches_unbatched_cache(self, model):
        """Single-item batch cache should match unbatched cache."""
        model.model.eval()
        task = "Describe the image."
        image = generate_checkered_image(width=56, height=56, seed=7)

        single_inputs = model.prepare_messages(task, image)
        with model.run_with_cache(["lm_resid_post"]):
            model.forward(single_inputs)
        single_cache = model.cache[("lm_resid_post", 0)]

        with model.run_with_cache(["lm_resid_post"]):
            model.forward_batch(tasks=[task], images=[image])
        batch_cache = model.cache[("lm_resid_post", 0)]

        batch_on_device = batch_cache.to(single_cache.device)
        assert torch.allclose(single_cache, batch_on_device, atol=5e-3)
