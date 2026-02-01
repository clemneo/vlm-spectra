"""Shared cache hook test suites for HookedVLM models.

Each suite expects a ``vlm_model`` fixture that yields a loaded ``HookedVLM``
instance. The shapes suite additionally relies on a ``cache_all_hooks``
fixture that returns metadata produced by :func:`collect_cache_metadata`.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


__all__ = [
    "generate_test_image",
    "collect_cache_metadata",
    "CacheHookRegistrationSuite",
    "CacheHookShapesSuite",
    "CacheHookNonInvasivenessSuite",
    "CacheHookCleanupSuite",
]


def generate_test_image(width: int = 56, height: int = 56, seed: int | None = None):
    """Generate a deterministic RGB image for hook tests."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def collect_cache_metadata(model):
    """Run a forward pass with all cache hooks registered.

    Returns a dictionary with cached tensors plus relevant metadata so shape
    assertions can be shared between integration and acceptance tests.
    """
    model.model.eval()
    image = generate_test_image(seed=1)
    inputs = model.prepare_messages("Describe.", image)
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

    with model.run_with_cache(hooks) as cache:
        model.forward(inputs)

    return {
        "cache": cache,
        "seq_len": seq_len,
        "num_layers": model.adapter.lm_num_layers,
        "hidden_dim": model.adapter.lm_hidden_dim,
        "num_heads": model.adapter.lm_num_heads,
        "num_kv_heads": model.adapter.lm_num_kv_heads,
        "head_dim": model.adapter.lm_head_dim,
        "mlp_dim": model.adapter.lm_mlp_dim,
    }


class CacheHookRegistrationSuite:
    """Tests focused on cache hook registration mechanics."""

    def test_hook_registers_on_module(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            vlm_model.forward(inputs)

        assert vlm_model.cache is not None
        assert len(vlm_model.cache.keys()) > 0

    def test_hook_fires_on_forward(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            vlm_model.forward(inputs)

        num_cached = len([k for k in vlm_model.cache.keys() if "hook_resid_post" in k])
        assert num_cached == vlm_model.adapter.lm_num_layers

    def test_multiple_hooks_same_module(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        hooks = ["lm.blocks.*.hook_resid_post", "lm.blocks.*.mlp.hook_out"]
        with vlm_model.run_with_cache(hooks):
            vlm_model.forward(inputs)

        resid_count = len([k for k in vlm_model.cache.keys() if "hook_resid_post" in k])
        mlp_count = len([k for k in vlm_model.cache.keys() if "mlp.hook_out" in k])

        assert resid_count == vlm_model.adapter.lm_num_layers
        assert mlp_count == vlm_model.adapter.lm_num_layers


class CacheHookShapesSuite:
    """Tests that cached activations have the expected shapes."""

    def test_hook_resid_pre_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.hook_resid_pre"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_hook_resid_pre_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.hook_resid_pre").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["hidden_dim"],
        )

    def test_hook_resid_post_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.hook_resid_post"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_hook_resid_post_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.hook_resid_post").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["hidden_dim"],
        )

    def test_attn_hook_in_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_in"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_in_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_in").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["hidden_dim"],
        )

    def test_attn_hook_q_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_q"].shape == (
            1,
            c["seq_len"],
            c["num_heads"],
            c["head_dim"],
        )

    def test_attn_hook_q_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_q").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["num_heads"],
            c["head_dim"],
        )

    def test_attn_hook_k_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_k"].shape == (
            1,
            c["seq_len"],
            c["num_kv_heads"],
            c["head_dim"],
        )

    def test_attn_hook_k_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_k").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["num_kv_heads"],
            c["head_dim"],
        )

    def test_attn_hook_v_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_v"].shape == (
            1,
            c["seq_len"],
            c["num_kv_heads"],
            c["head_dim"],
        )

    def test_attn_hook_v_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_v").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["num_kv_heads"],
            c["head_dim"],
        )

    def test_attn_hook_z_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_z"].shape == (
            1,
            c["seq_len"],
            c["num_heads"],
            c["head_dim"],
        )

    def test_attn_hook_z_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_z").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["num_heads"],
            c["head_dim"],
        )

    def test_attn_hook_out_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_out"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_attn_hook_out_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_out").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["hidden_dim"],
        )

    def test_attn_hook_pattern_sums_to_one(self, cache_all_hooks):
        c = cache_all_hooks
        patterns = c["cache"].stack("lm.blocks.*.attn.hook_pattern")
        sums = patterns.float().sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=5e-3, rtol=5e-3)

    def test_attn_hook_scores_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_scores"].shape == (
            1,
            c["num_heads"],
            c["seq_len"],
            c["seq_len"],
        )

    def test_attn_hook_scores_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_scores").shape == (
            c["num_layers"],
            1,
            c["num_heads"],
            c["seq_len"],
            c["seq_len"],
        )

    def test_attn_hook_head_out_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.attn.hook_head_out"].shape == (
            1,
            c["seq_len"],
            c["num_heads"],
            c["hidden_dim"],
        )

    def test_attn_hook_head_out_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.attn.hook_head_out").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["num_heads"],
            c["hidden_dim"],
        )

    def test_mlp_hook_in_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_in"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_mlp_hook_in_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_in").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["hidden_dim"],
        )

    def test_mlp_hook_out_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_out"].shape == (1, c["seq_len"], c["hidden_dim"])

    def test_mlp_hook_out_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_out").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["hidden_dim"],
        )

    def test_mlp_hook_pre_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_pre"].shape == (1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_pre_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_pre").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["mlp_dim"],
        )

    def test_mlp_hook_pre_linear_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_pre_linear"].shape == (1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_pre_linear_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_pre_linear").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["mlp_dim"],
        )

    def test_mlp_hook_post_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"]["lm.blocks.0.mlp.hook_post"].shape == (1, c["seq_len"], c["mlp_dim"])

    def test_mlp_hook_post_stacked_shape(self, cache_all_hooks):
        c = cache_all_hooks
        assert c["cache"].stack("lm.blocks.*.mlp.hook_post").shape == (
            c["num_layers"],
            1,
            c["seq_len"],
            c["mlp_dim"],
        )


class CacheHookNonInvasivenessSuite:
    """Tests that cache hooks do not alter model behavior."""

    def test_resid_post_equals_next_resid_pre(self, vlm_model):
        image = generate_test_image(seed=42)
        inputs = vlm_model.prepare_messages("Describe.", image)

        hooks = ["lm.blocks.*.hook_resid_pre", "lm.blocks.*.hook_resid_post"]
        with vlm_model.run_with_cache(hooks) as cache:
            vlm_model.forward(inputs)

        num_layers = vlm_model.adapter.lm_num_layers
        for i in range(num_layers - 1):
            resid_post = cache[f"lm.blocks.{i}.hook_resid_post"]
            resid_pre_next = cache[f"lm.blocks.{i + 1}.hook_resid_pre"]
            assert torch.allclose(
                resid_post,
                resid_pre_next,
                atol=0,
                rtol=0,
            ), f"resid_post of layer {i} should equal resid_pre of layer {i + 1}"

    def test_logits_identical_with_and_without_cache(self, vlm_model):
        image = generate_test_image(seed=42)
        inputs = vlm_model.prepare_messages("Describe.", image)

        output_no_hooks = vlm_model.forward(inputs)
        logits_no_hooks = output_no_hooks.logits.clone()

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
        with vlm_model.run_with_cache(hooks):
            output_with_hooks = vlm_model.forward(inputs)
        logits_with_hooks = output_with_hooks.logits

        assert torch.allclose(logits_no_hooks, logits_with_hooks, atol=0, rtol=0)


class CacheHookCleanupSuite:
    """Tests covering cache hook cleanup behavior."""

    def test_hooks_removed_after_context_exit(self, vlm_model):
        image = generate_test_image(seed=42)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            vlm_model.forward(inputs)

        assert len(vlm_model._hook_manager._cache_handles) == 0
        assert len(vlm_model._hook_manager._registered_hooks) == 0

    def test_cache_accessible_after_context_exit(self, vlm_model):
        image = generate_test_image(seed=42)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]) as cache:
            vlm_model.forward(inputs)

        assert len(cache.keys()) > 0
        assert "lm.blocks.0.hook_resid_post" in cache

    def test_repeated_cache_runs_no_hook_accumulation(self, vlm_model):
        image = generate_test_image(seed=42)
        inputs = vlm_model.prepare_messages("Describe.", image)

        for _ in range(2):
            with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
                vlm_model.forward(inputs)

        assert len(vlm_model._hook_manager._cache_handles) == 0

    def test_forward_works_after_cache_context(self, vlm_model):
        image = generate_test_image(seed=42)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            vlm_model.forward(inputs)

        output = vlm_model.forward(inputs)
        assert output.logits is not None
