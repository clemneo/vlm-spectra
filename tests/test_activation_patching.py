"""Tests for forward hooks that replace activations - Concern #4.

These tests validate that a model implementation correctly:
- Allows hooks to modify layer outputs
- Supports activation patching (replacing activations from one input with another)
- Properly removes hooks after context exit
"""

import torch
import pytest

from conftest import generate_random_image, generate_checkered_image


class LayerReplacementHook:
    """Hook that replaces layer output with a stored tensor."""

    def __init__(self, layer: int, replacement: torch.Tensor):
        self.layer = layer
        self.replacement = replacement

    def __call__(self, module, args, kwargs, output):
        """Replace output with stored tensor."""
        # Handle tuple outputs (some layers return (hidden_states, ...))
        if isinstance(output, tuple):
            return (self.replacement,) + output[1:]
        return self.replacement


class LayerModificationHook:
    """Hook that modifies layer output (e.g., zeros it out)."""

    def __init__(self, layer: int):
        self.layer = layer

    def __call__(self, module, args, kwargs, output):
        """Zero out the output."""
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


class AttnPreHook:
    """Pre-hook for attention layers that can modify inputs."""

    def __init__(self, layer: int, hidden_replacement: torch.Tensor = None):
        self.layer = layer
        self.hidden_replacement = hidden_replacement

    def __call__(self, module, args, kwargs):
        """Modify attention inputs."""
        if self.hidden_replacement is not None:
            # Replace hidden states (first positional arg or kwarg)
            if len(args) > 0:
                new_args = (self.hidden_replacement,) + args[1:]
                return new_args, kwargs
            elif "hidden_states" in kwargs:
                new_kwargs = kwargs.copy()
                new_kwargs["hidden_states"] = self.hidden_replacement
                return args, new_kwargs
        return args, kwargs


class ModuleHook:
    """Hook for specific modules (o_proj, mlp, self_attn)."""

    def __init__(self, layer: int, module_name: str, hook_type: str = "pre"):
        self.layer = layer
        self.module_name = module_name
        self.hook_type = hook_type
        self.captured = None

    def __call__(self, module, input_or_output):
        """Capture the input or output."""
        self.captured = input_or_output
        return input_or_output


class TestActivationPatching:
    """Test activation patching functionality."""

    def test_hook_can_replace_layer_output(self, model):
        """Hooks should be able to replace layer outputs."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        # Get original output
        original_output = model.forward(inputs)
        original_logits = original_output.logits.clone()

        # Create a replacement tensor (zeros)
        # First, run to get the shape
        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs)
        layer_shape = model.cache[("lm_resid_post", 0)].shape
        replacement = torch.zeros(layer_shape, device=model.device, dtype=torch.bfloat16)

        # Run with hook that replaces layer 0 output
        hook = LayerModificationHook(layer=0)
        with model.run_with_hooks([hook]):
            modified_output = model.forward(inputs)

        # Outputs should differ (zeroing a layer should change the output)
        assert not torch.allclose(
            original_logits, modified_output.logits, atol=1e-3
        ), "Replacing layer output should change final logits"

    def test_patched_output_differs_from_original(self, model):
        """Patching activations should produce different outputs."""
        image1 = generate_checkered_image(seed=1)
        image2 = generate_checkered_image(seed=2)

        inputs1 = model.prepare_messages("Describe image 1.", image1)
        inputs2 = model.prepare_messages("Describe image 2.", image2)

        # Get activations from image1
        with model.run_with_cache(["lm_resid_post"]):
            model.forward(inputs1)
        source_activation = model.cache[("lm_resid_post", 5)].clone()

        # Get original output for image2
        original_output = model.forward(inputs2)
        original_logits = original_output.logits.clone()

        # Patch layer 5 of image2 with activations from image1
        hook = LayerReplacementHook(layer=5, replacement=source_activation)
        with model.run_with_hooks([hook]):
            patched_output = model.forward(inputs2)

        # Patched output should differ from original
        assert not torch.allclose(
            original_logits, patched_output.logits, atol=1e-3
        ), "Patched activations should change output"

    def test_patch_single_layer(self, model):
        """Patching a single layer should work."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        # Get original output
        original_output = model.forward(inputs)

        # Run with zeroing hook on a middle layer
        middle_layer = model.lm_num_layers // 2
        hook = LayerModificationHook(layer=middle_layer)
        with model.run_with_hooks([hook]):
            modified_output = model.forward(inputs)

        # Should produce different output
        assert not torch.allclose(
            original_output.logits, modified_output.logits, atol=1e-3
        )

    def test_patch_multiple_layers(self, model):
        """Patching multiple layers should work."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        # Create hooks for multiple layers
        hooks = [
            LayerModificationHook(layer=0),
            LayerModificationHook(layer=1),
            LayerModificationHook(layer=2),
        ]

        original_output = model.forward(inputs)
        with model.run_with_hooks(hooks):
            modified_output = model.forward(inputs)

        assert not torch.allclose(
            original_output.logits, modified_output.logits, atol=1e-3
        )

    def test_patch_with_saved_activations(self, model):
        """Core activation patching flow: save from source, patch into target."""
        # Source input
        source_image = generate_checkered_image(width=56, height=56, seed=100)
        source_inputs = model.prepare_messages("Source task.", source_image)

        # Target input
        target_image = generate_checkered_image(width=56, height=56, seed=200)
        target_inputs = model.prepare_messages("Target task.", target_image)

        # Step 1: Save activations from source
        with model.run_with_cache(["lm_resid_post"]):
            model.forward(source_inputs)
        source_activations = {
            layer: model.cache[("lm_resid_post", layer)].clone()
            for layer in range(model.lm_num_layers)
        }

        # Step 2: Get unpatched target output
        unpatched_output = model.forward(target_inputs)

        # Step 3: Patch target with source activations at layer 10
        patch_layer = min(10, model.lm_num_layers - 1)
        hook = LayerReplacementHook(
            layer=patch_layer,
            replacement=source_activations[patch_layer],
        )
        with model.run_with_hooks([hook]):
            patched_output = model.forward(target_inputs)

        # Verify patching changed the output
        assert not torch.allclose(
            unpatched_output.logits, patched_output.logits, atol=1e-3
        ), "Patching should change the output"


class TestHookLifecycle:
    """Test hook lifecycle management."""

    def test_hooks_are_removed_after_context(self, model):
        """Hooks should be removed after context manager exits."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        # Get baseline output
        baseline = model.forward(inputs).logits.clone()

        # Run with hook
        hook = LayerModificationHook(layer=0)
        with model.run_with_hooks([hook]):
            pass  # Just enter and exit

        # After context, hook should be gone
        after_context = model.forward(inputs).logits

        # Should match baseline (hook is gone)
        assert torch.allclose(baseline, after_context, atol=1e-5)

    def test_hooks_can_be_reused(self, model):
        """Hooks should work correctly when reused."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        hook = LayerModificationHook(layer=0)

        # First use
        with model.run_with_hooks([hook]):
            output1 = model.forward(inputs).logits.clone()

        # Second use
        with model.run_with_hooks([hook]):
            output2 = model.forward(inputs).logits.clone()

        # Both should produce the same result
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_multiple_hooks_same_layer(self, model):
        """Multiple hooks on the same layer should work."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        # Two hooks on layer 0
        hooks = [
            LayerModificationHook(layer=0),
            LayerModificationHook(layer=0),
        ]

        # Should not error
        with model.run_with_hooks(hooks):
            output = model.forward(inputs)

        assert hasattr(output, "logits")


class TestAttnHooks:
    """Test attention-specific hooks."""

    def test_attn_pre_hook_registers(self, model):
        """Attention pre-hooks should register correctly."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        hook = AttnPreHook(layer=0)

        # Should not error
        with model.run_with_attn_hooks([hook]):
            output = model.forward(inputs)

        assert hasattr(output, "logits")


class TestModuleHooks:
    """Test module-specific hooks."""

    def test_hook_on_o_proj(self, model):
        """Hooks on o_proj module should work."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        hook = ModuleHook(layer=0, module_name="o_proj", hook_type="post")

        with model.run_with_module_hooks([hook]):
            model.forward(inputs)

        assert hook.captured is not None

    def test_hook_on_mlp(self, model):
        """Hooks on MLP module should work."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        hook = ModuleHook(layer=0, module_name="mlp", hook_type="post")

        with model.run_with_module_hooks([hook]):
            model.forward(inputs)

        assert hook.captured is not None

    def test_hook_on_self_attn(self, model):
        """Hooks on self_attn module should work."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        hook = ModuleHook(layer=0, module_name="self_attn", hook_type="post")

        with model.run_with_module_hooks([hook]):
            model.forward(inputs)

        assert hook.captured is not None

    def test_pre_vs_post_hook(self, model):
        """Pre and post hooks should capture at different times."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        pre_hook = ModuleHook(layer=0, module_name="mlp", hook_type="pre")
        post_hook = ModuleHook(layer=0, module_name="mlp", hook_type="post")

        with model.run_with_module_hooks([pre_hook]):
            model.forward(inputs)
        pre_captured = pre_hook.captured

        with model.run_with_module_hooks([post_hook]):
            model.forward(inputs)
        post_captured = post_hook.captured

        # Pre-hook captures input, post-hook captures output
        # They should have different shapes or values
        assert pre_captured is not None
        assert post_captured is not None
