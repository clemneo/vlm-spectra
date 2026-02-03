"""Shared patch hook test suites for HookedVLM models.

These suites expect a ``vlm_model`` fixture that yields a loaded ``HookedVLM``
instance to run tuple-based patch hook tests against.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from vlm_spectra.core.patch_hooks import (
    VALID_PATCH_HOOK_TYPES,
    VALID_PRE_PATCH_HOOK_TYPES,
    validate_patch_hook_type,
    PatchActivation,
    ZeroAblation,
    AddActivation,
    ScaleActivation,
    PatchHead,
)

__all__ = [
    "generate_test_image",
    "PatchTupleAPISuite",
    "PatchValidHookPointsSuite",
    "PatchValidateHookTypeSuite",
    "PatchHelperClassesSuite",
    "PatchHookCleanupSuite",
    "PatchCacheInteractionSuite",
    "PatchPreHookSuite",
    "PatchPreHookCleanupSuite",
]


def generate_test_image(width: int = 56, height: int = 56, seed: int | None = None):
    """Generate a deterministic RGB image for patch hook tests."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class PatchTupleAPISuite:
    """Tests covering the tuple-based patch hook API."""

    def test_function_hook_modifies_output(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_lambda_hook(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        with vlm_model.run_with_hooks([
            ("lm.blocks.0.hook_resid_post", lambda m, a, k, o: torch.zeros_like(o))
        ]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_hook_none_return_preserves_activation(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        def passthrough_hook(module, args, kwargs, output):
            return None

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", passthrough_hook)]):
            modified = vlm_model.forward(inputs).logits

        assert torch.allclose(original, modified, atol=1e-5)

    def test_wildcard_pattern_applies_to_all_layers(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        hooked_layers = []

        def tracking_hook(module, args, kwargs, output):
            hooked_layers.append(True)
            return None

        with vlm_model.run_with_hooks([("lm.blocks.*.hook_resid_post", tracking_hook)]):
            vlm_model.forward(inputs)

        assert len(hooked_layers) == vlm_model.lm_num_layers

    def test_multiple_hooks_on_different_layers(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        hook_calls = {"layer_0": 0, "layer_1": 0}

        def hook_layer_0(module, args, kwargs, output):
            hook_calls["layer_0"] += 1
            return None

        def hook_layer_1(module, args, kwargs, output):
            hook_calls["layer_1"] += 1
            return None

        with vlm_model.run_with_hooks([
            ("lm.blocks.0.hook_resid_post", hook_layer_0),
            ("lm.blocks.1.hook_resid_post", hook_layer_1),
        ]):
            vlm_model.forward(inputs)

        assert hook_calls["layer_0"] == 1
        assert hook_calls["layer_1"] == 1


class PatchValidHookPointsSuite:
    """Tests validating supported hook points."""

    def test_valid_hook_resid_post(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_hooks([
            ("lm.blocks.0.hook_resid_post", lambda m, a, k, o: None)
        ]):
            vlm_model.forward(inputs)

    def test_valid_attn_hook_out(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_hooks([
            ("lm.blocks.0.attn.hook_out", lambda m, a, k, o: None)
        ]):
            vlm_model.forward(inputs)

    def test_valid_mlp_hook_out(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_hooks([
            ("lm.blocks.0.mlp.hook_out", lambda m, a, k, o: None)
        ]):
            vlm_model.forward(inputs)

    def test_invalid_pre_hook_rejected(self, vlm_model):
        with pytest.raises(ValueError, match="pre-hook"):
            with vlm_model.run_with_hooks([
                ("lm.blocks.0.hook_resid_pre", lambda m, a, k, o: None)
            ]):
                pass

    def test_invalid_virtual_hook_rejected(self, vlm_model):
        with pytest.raises(ValueError, match="virtual"):
            with vlm_model.run_with_hooks([
                ("lm.blocks.0.attn.hook_pattern", lambda m, a, k, o: None)
            ]):
                pass

    def test_invalid_hook_type_rejected(self, vlm_model):
        with pytest.raises(ValueError):
            with vlm_model.run_with_hooks([
                ("lm.blocks.0.invalid_hook", lambda m, a, k, o: None)
            ]):
                pass


class PatchValidateHookTypeSuite:
    """Tests for :func:`validate_patch_hook_type`."""

    def test_valid_post_hook_types_return_false(self):
        """Post-hooks should return False (not a pre-hook)."""
        for hook_type in VALID_PATCH_HOOK_TYPES:
            assert validate_patch_hook_type(hook_type) is False

    def test_valid_pre_hook_types_return_true(self):
        """Pre-hooks in VALID_PRE_PATCH_HOOK_TYPES should return True."""
        for hook_type in VALID_PRE_PATCH_HOOK_TYPES:
            assert validate_patch_hook_type(hook_type) is True

    def test_unsupported_pre_hook_raises(self):
        with pytest.raises(ValueError, match="pre-hook"):
            validate_patch_hook_type("hook_resid_pre")

        with pytest.raises(ValueError, match="pre-hook"):
            validate_patch_hook_type("attn.hook_in")

    def test_virtual_hook_raises(self):
        with pytest.raises(ValueError, match="virtual"):
            validate_patch_hook_type("attn.hook_pattern")

        with pytest.raises(ValueError, match="virtual"):
            validate_patch_hook_type("attn.hook_head_out")

    def test_unknown_hook_raises(self):
        with pytest.raises(ValueError, match="Unknown hook type"):
            validate_patch_hook_type("totally_made_up")


class PatchHelperClassesSuite:
    """Tests around helper classes that implement hook callables."""

    def test_patch_activation(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        hidden_dim = vlm_model.adapter.lm_hidden_dim
        replacement = torch.zeros(hidden_dim, device=vlm_model.device)

        hook = PatchActivation(replacement, token_idx=-1)

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_zero_ablation(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        hook = ZeroAblation(token_idx=-1)

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_add_activation(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        hidden_dim = vlm_model.adapter.lm_hidden_dim
        direction = torch.randn(hidden_dim, device=vlm_model.device) * 10

        hook = AddActivation(direction, scale=1.0, token_idx=-1)

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_scale_activation(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        hook = ScaleActivation(scale=0.0, token_idx=-1)

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)


class PatchHookCleanupSuite:
    """Tests verifying hook cleanup behavior."""

    def test_hooks_removed_after_context(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        baseline = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
            pass

        after = vlm_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)

    def test_hooks_removed_on_exception(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        baseline = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with pytest.raises(RuntimeError):
            with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
                raise RuntimeError("Test exception")

        after = vlm_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)


class PatchCacheInteractionSuite:
    """Tests covering interaction between cache and patch hooks."""

    def test_cache_and_hooks_nest(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        baseline = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with vlm_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
                outputs = vlm_model.forward(inputs)

        assert vlm_model.cache is not None
        assert "lm.blocks.0.hook_resid_post" in vlm_model.cache
        assert not torch.allclose(baseline, outputs.logits, atol=1e-3)

    def test_cache_sees_modified_activations(self, vlm_model):
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        with vlm_model.run_with_cache(["lm.blocks.0.hook_resid_post"]):
            vlm_model.forward(inputs)
        unmodified = vlm_model.cache["lm.blocks.0.hook_resid_post"].clone()

        scale_hook = ScaleActivation(scale=2.0)

        with vlm_model.run_with_cache(["lm.blocks.0.hook_resid_post"]):
            with vlm_model.run_with_hooks([("lm.blocks.0.hook_resid_post", scale_hook)]):
                vlm_model.forward(inputs)

        modified = vlm_model.cache["lm.blocks.0.hook_resid_post"]
        assert torch.allclose(modified, unmodified * 2, atol=1e-3)


class PatchPreHookSuite:
    """Tests for pre-hook patching support (attn.hook_z)."""

    def test_attn_hook_z_modifies_output(self, vlm_model):
        """Verify patching attn.hook_z changes model output."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with vlm_model.run_with_hooks([("lm.blocks.0.attn.hook_z", zero_hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_pre_hook_none_return_preserves_activation(self, vlm_model):
        """Verify None passthrough doesn't modify activations."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        def passthrough_hook(module, args, kwargs, output):
            return None

        with vlm_model.run_with_hooks([("lm.blocks.0.attn.hook_z", passthrough_hook)]):
            modified = vlm_model.forward(inputs).logits

        assert torch.allclose(original, modified, atol=1e-5)

    def test_patch_head_helper_with_hook_z(self, vlm_model):
        """Verify PatchHead helper works with attn.hook_z."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        num_heads = vlm_model.adapter.lm_num_heads
        head_dim = vlm_model.adapter.lm_head_dim
        replacement = torch.zeros(head_dim, device=vlm_model.device)

        hook = PatchHead(head_idx=0, replacement=replacement, num_heads=num_heads)

        with vlm_model.run_with_hooks([("lm.blocks.0.attn.hook_z", hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_zero_ablation_with_pre_hook(self, vlm_model):
        """Verify ZeroAblation works with pre-hooks."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        original = vlm_model.forward(inputs).logits.clone()

        hook = ZeroAblation(token_idx=-1)

        with vlm_model.run_with_hooks([("lm.blocks.0.attn.hook_z", hook)]):
            modified = vlm_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_wildcard_pre_hook_applies_to_all_layers(self, vlm_model):
        """Verify wildcards work with pre-hooks."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        hooked_layers = []

        def tracking_hook(module, args, kwargs, output):
            hooked_layers.append(True)
            return None

        with vlm_model.run_with_hooks([("lm.blocks.*.attn.hook_z", tracking_hook)]):
            vlm_model.forward(inputs)

        assert len(hooked_layers) == vlm_model.lm_num_layers

    def test_unsupported_pre_hook_rejected(self, vlm_model):
        """Verify unsupported pre-hooks are still rejected."""
        with pytest.raises(ValueError, match="pre-hook"):
            with vlm_model.run_with_hooks([
                ("lm.blocks.0.hook_resid_pre", lambda m, a, k, o: None)
            ]):
                pass

        with pytest.raises(ValueError, match="pre-hook"):
            with vlm_model.run_with_hooks([
                ("lm.blocks.0.attn.hook_in", lambda m, a, k, o: None)
            ]):
                pass

        with pytest.raises(ValueError, match="pre-hook"):
            with vlm_model.run_with_hooks([
                ("lm.blocks.0.mlp.hook_post", lambda m, a, k, o: None)
            ]):
                pass


class PatchPreHookCleanupSuite:
    """Tests verifying pre-hook cleanup behavior."""

    def test_pre_hooks_removed_after_context(self, vlm_model):
        """Verify pre-hooks are removed after context exits."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        baseline = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with vlm_model.run_with_hooks([("lm.blocks.0.attn.hook_z", zero_hook)]):
            pass

        after = vlm_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)

    def test_pre_hooks_removed_on_exception(self, vlm_model):
        """Verify pre-hooks are removed even when exception occurs."""
        image = generate_test_image(seed=1)
        inputs = vlm_model.prepare_messages("Describe.", image)

        baseline = vlm_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with pytest.raises(RuntimeError):
            with vlm_model.run_with_hooks([("lm.blocks.0.attn.hook_z", zero_hook)]):
                raise RuntimeError("Test exception")

        after = vlm_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)
