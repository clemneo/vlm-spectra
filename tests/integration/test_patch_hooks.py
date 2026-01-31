"""Integration tests for the tuple-based patch hook API.

These tests verify the new run_with_hooks API that uses (hook_name, hook_fn) tuples
instead of Hook objects. Tests cover:
- Basic tuple-based API usage
- Valid/invalid hook point validation
- Helper class functionality
- Hook cleanup on context exit
- Interaction between cache and patch hooks
"""

import pytest
import numpy as np
import torch
from PIL import Image

from vlm_spectra.core.patch_hooks import (
    VALID_PATCH_HOOK_TYPES,
    validate_patch_hook_type,
    PatchActivation,
    ZeroAblation,
    AddActivation,
    ScaleActivation,
)


def generate_test_image(width=56, height=56, seed=None):
    """Generate a simple test image."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestTupleBasedAPI:
    """Test the new tuple-based hook API."""

    def test_function_hook_modifies_output(self, tiny_model):
        """A function hook should modify layer output."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
            modified = tiny_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_lambda_hook(self, tiny_model):
        """Lambda functions should work as hook_fn."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        with tiny_model.run_with_hooks([
            ("lm.blocks.0.hook_resid_post", lambda m, a, k, o: torch.zeros_like(o))
        ]):
            modified = tiny_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_hook_none_return_preserves_activation(self, tiny_model):
        """Returning None from hook should preserve original activation."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        def passthrough_hook(module, args, kwargs, output):
            return None  # Signal to keep original

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", passthrough_hook)]):
            modified = tiny_model.forward(inputs).logits

        assert torch.allclose(original, modified, atol=1e-5)

    def test_wildcard_pattern_applies_to_all_layers(self, tiny_model):
        """Wildcard pattern should apply hook to all layers."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Track which layers get hooked
        hooked_layers = []

        def tracking_hook(module, args, kwargs, output):
            hooked_layers.append(True)
            return None  # Don't modify

        with tiny_model.run_with_hooks([("lm.blocks.*.hook_resid_post", tracking_hook)]):
            tiny_model.forward(inputs)

        # Should have hooked all layers
        assert len(hooked_layers) == tiny_model.lm_num_layers

    def test_multiple_hooks_on_different_layers(self, tiny_model):
        """Multiple hooks on different layers should all fire."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        hook_calls = {"layer_0": 0, "layer_1": 0}

        def hook_layer_0(module, args, kwargs, output):
            hook_calls["layer_0"] += 1
            return None

        def hook_layer_1(module, args, kwargs, output):
            hook_calls["layer_1"] += 1
            return None

        with tiny_model.run_with_hooks([
            ("lm.blocks.0.hook_resid_post", hook_layer_0),
            ("lm.blocks.1.hook_resid_post", hook_layer_1),
        ]):
            tiny_model.forward(inputs)

        assert hook_calls["layer_0"] == 1
        assert hook_calls["layer_1"] == 1


class TestValidHookPoints:
    """Test validation of hook points."""

    def test_valid_hook_resid_post(self, tiny_model):
        """hook_resid_post should be valid for patching."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Should not raise
        with tiny_model.run_with_hooks([
            ("lm.blocks.0.hook_resid_post", lambda m, a, k, o: None)
        ]):
            tiny_model.forward(inputs)

    def test_valid_attn_hook_out(self, tiny_model):
        """attn.hook_out should be valid for patching."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Should not raise
        with tiny_model.run_with_hooks([
            ("lm.blocks.0.attn.hook_out", lambda m, a, k, o: None)
        ]):
            tiny_model.forward(inputs)

    def test_valid_mlp_hook_out(self, tiny_model):
        """mlp.hook_out should be valid for patching."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Should not raise
        with tiny_model.run_with_hooks([
            ("lm.blocks.0.mlp.hook_out", lambda m, a, k, o: None)
        ]):
            tiny_model.forward(inputs)

    def test_invalid_pre_hook_rejected(self, tiny_model):
        """Pre-hooks like hook_resid_pre should be rejected."""
        with pytest.raises(ValueError, match="pre-hook"):
            with tiny_model.run_with_hooks([
                ("lm.blocks.0.hook_resid_pre", lambda m, a, k, o: None)
            ]):
                pass

    def test_invalid_virtual_hook_rejected(self, tiny_model):
        """Virtual hooks like attn.hook_pattern should be rejected."""
        with pytest.raises(ValueError, match="virtual"):
            with tiny_model.run_with_hooks([
                ("lm.blocks.0.attn.hook_pattern", lambda m, a, k, o: None)
            ]):
                pass

    def test_invalid_hook_type_rejected(self, tiny_model):
        """Unknown hook types should be rejected."""
        with pytest.raises(ValueError):
            with tiny_model.run_with_hooks([
                ("lm.blocks.0.invalid_hook", lambda m, a, k, o: None)
            ]):
                pass


class TestValidatePatchHookType:
    """Test the validate_patch_hook_type function directly."""

    def test_valid_hook_types_pass(self):
        """All valid hook types should pass validation."""
        for hook_type in VALID_PATCH_HOOK_TYPES:
            # Should not raise
            validate_patch_hook_type(hook_type)

    def test_pre_hook_raises(self):
        """Pre-hooks should raise ValueError."""
        with pytest.raises(ValueError, match="pre-hook"):
            validate_patch_hook_type("hook_resid_pre")

        with pytest.raises(ValueError, match="pre-hook"):
            validate_patch_hook_type("attn.hook_in")

    def test_virtual_hook_raises(self):
        """Virtual hooks should raise ValueError."""
        with pytest.raises(ValueError, match="virtual"):
            validate_patch_hook_type("attn.hook_pattern")

        with pytest.raises(ValueError, match="virtual"):
            validate_patch_hook_type("attn.hook_head_out")

    def test_unknown_hook_raises(self):
        """Unknown hook types should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown hook type"):
            validate_patch_hook_type("totally_made_up")


class TestPatchHelperClasses:
    """Test the patch helper classes."""

    def test_patch_activation(self, tiny_model):
        """PatchActivation should replace activations."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        # Create a replacement tensor (zeros)
        # We'll patch at token_idx=-1 (last token)
        hidden_dim = tiny_model.adapter.lm_hidden_dim
        replacement = torch.zeros(hidden_dim, device=tiny_model.device)

        hook = PatchActivation(replacement, token_idx=-1)

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = tiny_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_zero_ablation(self, tiny_model):
        """ZeroAblation should zero out activations."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        hook = ZeroAblation(token_idx=-1)

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = tiny_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_add_activation(self, tiny_model):
        """AddActivation should add to activations."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        hidden_dim = tiny_model.adapter.lm_hidden_dim
        direction = torch.randn(hidden_dim, device=tiny_model.device) * 10

        hook = AddActivation(direction, scale=1.0, token_idx=-1)

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = tiny_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    def test_scale_activation(self, tiny_model):
        """ScaleActivation should scale activations."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        original = tiny_model.forward(inputs).logits.clone()

        # Scale by 0 to get maximum effect
        hook = ScaleActivation(scale=0.0, token_idx=-1)

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", hook)]):
            modified = tiny_model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)


class TestHookCleanup:
    """Test that hooks are properly cleaned up."""

    def test_hooks_removed_after_context(self, tiny_model):
        """Hooks should be removed after context exits."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        baseline = tiny_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
            pass  # Just enter and exit

        after = tiny_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)

    def test_hooks_removed_on_exception(self, tiny_model):
        """Hooks should be removed even if exception occurs."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        baseline = tiny_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with pytest.raises(RuntimeError):
            with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
                raise RuntimeError("Test exception")

        # Hooks should still be removed
        after = tiny_model.forward(inputs).logits
        assert torch.allclose(baseline, after, atol=1e-5)


class TestCacheAndHooksInteraction:
    """Test interaction between cache and patch hooks."""

    def test_cache_and_hooks_nest(self, tiny_model):
        """Cache and patch hooks should work together without interfering."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Get baseline logits for comparison
        baseline = tiny_model.forward(inputs).logits.clone()

        def zero_hook(module, args, kwargs, output):
            return torch.zeros_like(output)

        with tiny_model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", zero_hook)]):
                outputs = tiny_model.forward(inputs)

        # Cache should exist and have captured values
        assert tiny_model.cache is not None
        assert "lm.blocks.0.hook_resid_post" in tiny_model.cache

        # The patch hook should have affected the model output
        # (logits should be different from baseline due to zeroing layer 0)
        assert not torch.allclose(baseline, outputs.logits, atol=1e-3)

    def test_cache_sees_modified_activations(self, tiny_model):
        """Cache should see activations after patch hooks modify them."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        # Get unmodified activation for comparison
        with tiny_model.run_with_cache(["lm.blocks.0.hook_resid_post"]):
            tiny_model.forward(inputs)
        unmodified = tiny_model.cache["lm.blocks.0.hook_resid_post"].clone()

        # Now run with a scaling hook
        scale_hook = ScaleActivation(scale=2.0)

        with tiny_model.run_with_cache(["lm.blocks.0.hook_resid_post"]):
            with tiny_model.run_with_hooks([("lm.blocks.0.hook_resid_post", scale_hook)]):
                tiny_model.forward(inputs)

        modified = tiny_model.cache["lm.blocks.0.hook_resid_post"]

        # The modified activation should be approximately 2x the original
        assert torch.allclose(modified, unmodified * 2, atol=1e-3)
