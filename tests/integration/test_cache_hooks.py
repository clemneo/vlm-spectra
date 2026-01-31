"""Integration tests for cache hook registration mechanics.

These tests verify that cache hooks can be registered, fire correctly,
log shapes, and are properly cleaned up using SmolVLM-256M.
"""

import numpy as np
import pytest
import torch
from PIL import Image


def generate_test_image(width=56, height=56, seed=None):
    """Generate a simple test image."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestCacheHookRegistration:
    """Test cache hook registration on real modules."""

    def test_hook_registers_on_module(self, tiny_model):
        """Hook should register and return handle."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            tiny_model.forward(inputs)

        assert tiny_model.cache is not None
        assert len(tiny_model.cache.keys()) > 0
        print("cache keys", len(tiny_model.cache.keys()))

    def test_hook_fires_on_forward(self, tiny_model):
        """Hook callback should be invoked during forward pass."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            tiny_model.forward(inputs)

        # Should have captured activations for all layers
        num_cached = len([k for k in tiny_model.cache.keys() if "hook_resid_post" in k])
        print("cached hook_resid_post", num_cached)
        assert num_cached == tiny_model.adapter.lm_num_layers


    def test_multiple_hooks_same_module(self, tiny_model):
        """Multiple hooks on same module should coexist."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        hooks = ["lm.blocks.*.hook_resid_post", "lm.blocks.*.mlp.hook_out"]
        with tiny_model.run_with_cache(hooks):
            tiny_model.forward(inputs)

        resid_count = len([k for k in tiny_model.cache.keys() if "hook_resid_post" in k])
        mlp_count = len([k for k in tiny_model.cache.keys() if "mlp.hook_out" in k])
        print("cached hook_resid_post", resid_count)
        print("cached mlp.hook_out", mlp_count)

        assert resid_count == tiny_model.adapter.lm_num_layers
        assert mlp_count == tiny_model.adapter.lm_num_layers


@pytest.fixture(scope="class")
def cache_all_hooks(tiny_model):
    """Run forward pass once with all hooks and return cache + metadata."""
    image = generate_test_image(seed=1)
    inputs = tiny_model.prepare_messages("Describe.", image)
    seq_len = inputs["input_ids"].shape[1]

    hooks = [
        "lm.blocks.*.hook_resid_pre",
        "lm.blocks.*.hook_resid_post",
        "lm.blocks.*.attn.hook_in",
        "lm.blocks.*.attn.hook_q",
        "lm.blocks.*.attn.hook_k",
        "lm.blocks.*.attn.hook_v",
        "lm.blocks.*.attn.hook_z",
        "lm.blocks.*.attn.hook_out",
        "lm.blocks.*.attn.hook_pattern",
        "lm.blocks.*.attn.hook_scores",
        "lm.blocks.*.attn.hook_head_out",
        "lm.blocks.*.mlp.hook_in",
        "lm.blocks.*.mlp.hook_out",
        "lm.blocks.*.mlp.hook_pre",
        "lm.blocks.*.mlp.hook_pre_linear",
        "lm.blocks.*.mlp.hook_post",
    ]

    with tiny_model.run_with_cache(hooks) as cache:
        tiny_model.forward(inputs)

    return {
        "cache": cache,
        "seq_len": seq_len,
        "num_layers": tiny_model.adapter.lm_num_layers,
        "hidden_dim": tiny_model.adapter.lm_hidden_dim,
        "num_heads": tiny_model.adapter.lm_num_heads,
        "num_kv_heads": tiny_model.adapter.lm_num_kv_heads,
        "head_dim": tiny_model.adapter.lm_head_dim,
        "mlp_dim": tiny_model.adapter.lm_mlp_dim,
    }


class TestCacheHookShapes:
    """Test that cache hooks capture correct shapes."""

    def test_hook_resid_pre_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.hook_resid_pre"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_hook_resid_pre_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.hook_resid_pre").shape == (c["num_layers"], 1, c["seq_len"], c["hidden_dim"])

    def test_hook_resid_post_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.hook_resid_post"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_hook_resid_post_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.hook_resid_post").shape == (c["num_layers"], 1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_in_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_in"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_in_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_in").shape == (c["num_layers"], 1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_q_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_q"].shape == (1, c["seq_len"], c["num_heads"], c["head_dim"])

    def test_attn_hook_q_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_q").shape == (c["num_layers"], 1, c["seq_len"], c["num_heads"], c["head_dim"])

    def test_attn_hook_k_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_k"].shape == (1, c["seq_len"], c["num_kv_heads"], c["head_dim"])

    def test_attn_hook_k_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_k").shape == (c["num_layers"], 1, c["seq_len"], c["num_kv_heads"], c["head_dim"])

    def test_attn_hook_v_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_v"].shape == (1, c["seq_len"], c["num_kv_heads"], c["head_dim"])

    def test_attn_hook_v_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_v").shape == (c["num_layers"], 1, c["seq_len"], c["num_kv_heads"], c["head_dim"])

    def test_attn_hook_z_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_z"].shape == (1, c["seq_len"], c["num_heads"], c["head_dim"])

    def test_attn_hook_z_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_z").shape == (c["num_layers"], 1, c["seq_len"], c["num_heads"], c["head_dim"])

    def test_attn_hook_out_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_out"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_out_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_out").shape == (c["num_layers"], 1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_pattern_sums_to_one(self, cache_all_hooks):
        c = cache_all_hooks
        patterns = c["cache"].stack("lm.blocks.*.attn.hook_pattern")
        sums = patterns.float().sum(dim=-1)
        assert torch.allclose(
            sums,
            torch.ones_like(sums),
            atol=5e-3,
            rtol=5e-3,
        )

    def test_attn_hook_scores_shape(self, cache_all_hooks):
        c = cache_all_hooks
        # (batch, num_heads, seq_len, seq_len)
        assert c["cache"]["lm.blocks.0.attn.hook_scores"].shape == (1, c["num_heads"], c["seq_len"], c["seq_len"])

    def test_attn_hook_scores_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_scores").shape == (c["num_layers"], 1, c["num_heads"], c["seq_len"], c["seq_len"])

    def test_attn_hook_head_out_shape(self, cache_all_hooks):
        c = cache_all_hooks
        # (batch, seq_len, num_heads, hidden_dim)
        assert c["cache"]["lm.blocks.0.attn.hook_head_out"].shape == (1, c["seq_len"], c["num_heads"], c["hidden_dim"])

    def test_attn_hook_head_out_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_head_out").shape == (c["num_layers"], 1, c["seq_len"], c["num_heads"], c["hidden_dim"])

    def test_mlp_hook_in_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_in"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_mlp_hook_in_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_in").shape == (c["num_layers"], 1, c["seq_len"], c["hidden_dim"])

    def test_mlp_hook_out_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_out"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_mlp_hook_out_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_out").shape == (c["num_layers"], 1, c["seq_len"], c["hidden_dim"])

    def test_mlp_hook_pre_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_pre"].shape == (1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_pre_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_pre").shape == (c["num_layers"], 1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_pre_linear_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_pre_linear"].shape == (1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_pre_linear_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_pre_linear").shape == (c["num_layers"], 1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_post_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_post"].shape == (1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_post_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_post").shape == (c["num_layers"], 1, c["seq_len"], c["mlp_dim"])


class TestCacheNonInvasiveness:
    """Test that caching hooks don't affect model outputs."""

    def test_resid_post_equals_next_resid_pre(self, tiny_model):
        """Residual post of layer N should equal residual pre of layer N+1."""
        image = generate_test_image(seed=42)
        inputs = tiny_model.prepare_messages("Describe.", image)

        hooks = ["lm.blocks.*.hook_resid_pre", "lm.blocks.*.hook_resid_post"]
        with tiny_model.run_with_cache(hooks) as cache:
            tiny_model.forward(inputs)

        num_layers = tiny_model.adapter.lm_num_layers
        for i in range(num_layers - 1):
            resid_post = cache[f"lm.blocks.{i}.hook_resid_post"]
            resid_pre_next = cache[f"lm.blocks.{i + 1}.hook_resid_pre"]
            assert torch.allclose(resid_post, resid_pre_next, atol=0, rtol=0), \
                f"resid_post of layer {i} should equal resid_pre of layer {i + 1}"

    def test_logits_identical_with_and_without_cache(self, tiny_model):
        """Model output should be identical with and without caching."""
        image = generate_test_image(seed=42)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Forward without hooks
        output_no_hooks = tiny_model.forward(inputs)
        logits_no_hooks = output_no_hooks.logits.clone()

        # Forward with all hooks
        hooks = [
            "lm.blocks.*.hook_resid_pre",
            "lm.blocks.*.hook_resid_post",
            "lm.blocks.*.attn.hook_in",
            "lm.blocks.*.attn.hook_q",
            "lm.blocks.*.attn.hook_k",
            "lm.blocks.*.attn.hook_v",
            "lm.blocks.*.attn.hook_z",
            "lm.blocks.*.attn.hook_out",
            "lm.blocks.*.mlp.hook_in",
            "lm.blocks.*.mlp.hook_out",
        ]
        with tiny_model.run_with_cache(hooks):
            output_with_hooks = tiny_model.forward(inputs)
        logits_with_hooks = output_with_hooks.logits

        # Logits must be exactly equal
        assert torch.allclose(logits_no_hooks, logits_with_hooks, atol=0, rtol=0), \
            "Caching hooks should not modify model outputs"


class TestCacheCleanup:
    """Test that cache hooks are properly cleaned up."""

    def test_hooks_removed_after_context_exit(self, tiny_model):
        """Hook handles should be removed when context manager exits."""
        image = generate_test_image(seed=42)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            tiny_model.forward(inputs)

        # After context exit, internal hook tracking should be empty
        assert len(tiny_model._hook_manager._cache_handles) == 0
        assert len(tiny_model._hook_manager._registered_hooks) == 0

    def test_cache_accessible_after_context_exit(self, tiny_model):
        """Cache data should remain accessible after context exits."""
        image = generate_test_image(seed=42)
        inputs = tiny_model.prepare_messages("Describe.", image)

        with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]) as cache:
            tiny_model.forward(inputs)

        # Cache should still have data
        assert len(cache.keys()) > 0
        assert "lm.blocks.0.hook_resid_post" in cache

    def test_repeated_cache_runs_no_hook_accumulation(self, tiny_model):
        """Multiple cache runs should not accumulate hooks."""
        image = generate_test_image(seed=42)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Run cache twice
        for _ in range(2):
            with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
                tiny_model.forward(inputs)

        # Should have no leftover hooks
        assert len(tiny_model._hook_manager._cache_handles) == 0

    def test_forward_works_after_cache_context(self, tiny_model):
        """Forward should work normally after cache context exits."""
        image = generate_test_image(seed=42)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Run with cache first
        with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            tiny_model.forward(inputs)

        # Should be able to run forward again without issues
        output = tiny_model.forward(inputs)
        assert output.logits is not None
