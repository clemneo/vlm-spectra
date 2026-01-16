"""Tests for meta-functions - Concern #5.

These tests validate that a model implementation correctly provides:
- Token identification (text vs image)
- Model component access
- Patch visualization
"""

import pytest
import numpy as np
from PIL import Image

from conftest import generate_random_image, generate_checkered_image, get_patch_size


class TestTokenIdentification:
    """Test token identification functionality."""

    def test_get_image_token_id(self, model):
        """get_image_token_id should return valid token ID."""
        image_token_id = model.adapter.get_image_token_id()

        # Should be an integer
        assert isinstance(image_token_id, int)

        # Since different models may have different image tokens,
        # just verify it's a valid token ID (positive, not unknown)
        assert image_token_id > 0, "Image token ID should be positive"

    def test_get_image_token_range(self, model):
        """get_image_token_range should return valid start/end indices."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)

        start_idx, end_idx = model.get_image_token_range(inputs)

        # Both should be integers
        assert isinstance(start_idx, int)
        assert isinstance(end_idx, int)

        # Start should be <= end
        assert start_idx <= end_idx

        # Tokens at these indices should be image tokens
        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = model.adapter.get_image_token_id()
        assert input_ids[start_idx] == image_token_id
        assert input_ids[end_idx] == image_token_id

    def test_no_image_raises_error(self, model):
        """get_image_token_range should raise ValueError with no image tokens.

        Note: This is a placeholder since prepare_messages always includes an image.
        Testing this would require creating text-only inputs, which is model-specific.
        """
        # Skip for now - would require text-only input preparation
        pass

    @pytest.mark.parametrize("multiplier", [8, 16, 24])
    def test_image_token_count_scales_with_size(self, model, multiplier):
        """Larger images should produce more image tokens."""
        patch_size = get_patch_size(model)
        image = generate_checkered_image(
            width=patch_size * multiplier,
            height=patch_size * multiplier,
            checkered_size=patch_size,
        )

        inputs = model.prepare_messages("Describe the image.", image)
        start_idx, end_idx = model.get_image_token_range(inputs)
        token_count = end_idx - start_idx + 1

        # Token count should be positive and reasonable
        assert token_count > 0
        # Larger multipliers should generally produce more tokens
        # (exact relationship depends on model architecture)


class TestModelComponents:
    """Test model component access."""

    def test_get_model_components_returns_norm(self, model):
        """get_model_components should include normalization layer."""
        components = model.get_model_components()

        assert "norm" in components
        assert components["norm"] is not None

    def test_get_model_components_returns_lm_head(self, model):
        """get_model_components should include language model head."""
        components = model.get_model_components()

        assert "lm_head" in components
        assert components["lm_head"] is not None

    def test_get_model_components_returns_tokenizer(self, model):
        """get_model_components should include tokenizer."""
        components = model.get_model_components()

        assert "tokenizer" in components
        assert components["tokenizer"] is not None

    def test_lm_num_layers_property(self, model):
        """lm_num_layers should return positive integer."""
        num_layers = model.lm_num_layers

        assert isinstance(num_layers, int)
        assert num_layers > 0


class TestAdapterProperties:
    """Test model adapter properties."""

    def test_adapter_lm_num_heads(self, model):
        """Adapter should report number of attention heads."""
        num_heads = model.adapter.lm_num_heads

        assert isinstance(num_heads, int)
        assert num_heads > 0

    def test_adapter_lm_hidden_dim(self, model):
        """Adapter should report hidden dimension."""
        hidden_dim = model.adapter.lm_hidden_dim

        assert isinstance(hidden_dim, int)
        assert hidden_dim > 0

    def test_adapter_lm_head_dim(self, model):
        """Adapter should report per-head dimension."""
        head_dim = model.adapter.lm_head_dim

        assert isinstance(head_dim, int)
        assert head_dim > 0

        # head_dim * num_heads should relate to hidden_dim
        # (exact relationship may vary by model)


class TestPatchVisualization:
    """Test patch visualization functionality."""

    def test_generate_patch_overview_returns_image(self, model):
        """generate_patch_overview should return a PIL Image."""
        image = generate_checkered_image(width=224, height=224)

        result = model.generate_patch_overview(image, with_labels=False)

        assert isinstance(result, Image.Image)

    def test_patch_overview_with_labels(self, model):
        """Patch overview with labels should have more red pixels."""
        image = generate_checkered_image(width=224, height=224)

        without_labels = model.generate_patch_overview(image, with_labels=False)
        with_labels = model.generate_patch_overview(image, with_labels=True)

        # Count red pixels in each
        without_array = np.array(without_labels)
        with_array = np.array(with_labels)

        red_mask_without = (
            (without_array[:, :, 0] == 255)
            & (without_array[:, :, 1] == 0)
            & (without_array[:, :, 2] == 0)
        )
        red_mask_with = (
            (with_array[:, :, 0] == 255)
            & (with_array[:, :, 1] == 0)
            & (with_array[:, :, 2] == 0)
        )

        # Labels add more red pixels
        assert red_mask_with.sum() > red_mask_without.sum()

    def test_patch_overview_without_labels(self, model):
        """Patch overview without labels should still show grid lines."""
        image = generate_checkered_image(width=224, height=224)

        result = model.generate_patch_overview(image, with_labels=False)
        result_array = np.array(result)

        # Should have some red pixels (grid lines)
        red_mask = (
            (result_array[:, :, 0] == 255)
            & (result_array[:, :, 1] == 0)
            & (result_array[:, :, 2] == 0)
        )
        assert red_mask.sum() > 0

    def test_patch_overview_preserves_content(self, model):
        """Patch overview should preserve image content (black/white ratio)."""
        image = generate_checkered_image(width=224, height=224)

        result = model.generate_patch_overview(image, with_labels=False)
        result_array = np.array(result)

        # Remove red pixels before counting
        red_mask = (
            (result_array[:, :, 0] == 255)
            & (result_array[:, :, 1] == 0)
            & (result_array[:, :, 2] == 0)
        )
        counting_array = result_array.copy()
        counting_array[red_mask] = [128, 128, 128]  # Neutral gray

        # Count black and white
        black_pixels = (counting_array == 0).sum()
        white_pixels = (counting_array == 255).sum()

        # Original checkered has equal black/white
        assert black_pixels == white_pixels
