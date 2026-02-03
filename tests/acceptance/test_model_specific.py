"""Model-specific tests gated by capabilities.

These tests only run for models that have specific capabilities.
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


def generate_checkered_image(width=56, height=56, checkered_size=14, seed=None):
    """Generate a checkered pattern image."""
    _ = seed  # unused
    image = Image.new("RGB", (width, height))
    for x in range(0, width, checkered_size):
        for y in range(0, height, checkered_size):
            color = (
                (255, 255, 255)
                if (x + y) % (checkered_size * 2) < checkered_size
                else (0, 0, 0)
            )
            image.paste(color, (x, y, x + checkered_size, y + checkered_size))
    return image


class TestContiguousImageTokens:
    """Tests for models with contiguous image tokens."""

    @pytest.mark.requires_capability("contiguous_image_tokens")
    def test_image_tokens_contiguous(self, model):
        """Image tokens should be contiguous (no interleaved markers)."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)
        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = model.adapter.get_image_token_id()

        # Find all image token positions
        positions = (input_ids == image_token_id).nonzero().squeeze(-1)
        if len(positions) > 1:
            # Check they're consecutive
            diffs = positions[1:] - positions[:-1]
            assert (diffs == 1).all(), "Image tokens not contiguous"


class TestBatchedOperations:
    """Tests for models that support batching."""

    @pytest.mark.requires_capability("supports_batching")
    def test_batched_matches_unbatched(self, model):
        """Batched output should match single-item output."""
        model.model.eval()
        task = "Describe."
        image = generate_checkered_image(width=56, height=56, seed=42)

        # Single call
        single_inputs = model.prepare_messages(task, image)
        single_out = model.generate(single_inputs, max_new_tokens=5, do_sample=False)

        # Batch call
        batch_out = model.generate_batch(
            tasks=[task], images=[image], max_new_tokens=5, do_sample=False
        )

        tokenizer = model.processor.tokenizer
        single_text = tokenizer.decode(single_out.sequences[0], skip_special_tokens=True)
        batch_text = tokenizer.decode(batch_out.sequences[0], skip_special_tokens=True)

        assert single_text == batch_text

    @pytest.mark.requires_capability("supports_batching")
    def test_batch_forward_shape(self, model):
        """Batch forward should return correct batch dimension."""
        tasks = ["Describe image 1.", "Describe image 2."]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.forward_batch(tasks=tasks, images=images)

        assert outputs.logits.shape[0] == 2  # Batch size

    @pytest.mark.requires_capability("supports_batching")
    def test_batch_cache_shapes(self, model):
        """Batched cache should include batch dimension."""
        tasks = ["Describe image 1.", "Describe image 2."]
        images = [
            generate_checkered_image(width=56, height=56, seed=1),
            generate_checkered_image(width=56, height=56, seed=2),
        ]
        inputs = model.prepare_messages_batch(tasks, images)
        seq_len = inputs["input_ids"].shape[1]

        with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            model.forward_batch(tasks=tasks, images=images)

        sample = model.cache["lm.blocks.0.hook_resid_post"]
        expected_shape = (2, seq_len, model.adapter.lm_hidden_dim)
        assert sample.shape == expected_shape


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

    def test_lm_num_layers_property(self, model):
        """lm_num_layers should return positive integer."""
        num_layers = model.lm_num_layers

        assert isinstance(num_layers, int)
        assert num_layers > 0


class TestTokenIdentification:
    """Test token identification functionality."""

    def test_get_image_token_id(self, model):
        """get_image_token_id should return valid token ID."""
        image_token_id = model.adapter.get_image_token_id()

        assert isinstance(image_token_id, int)
        assert image_token_id > 0

    def test_image_tokens_in_input(self, model):
        """Processed input should contain image tokens."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)
        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = model.adapter.get_image_token_id()

        assert (input_ids == image_token_id).any()


class TestForwardGenerateConsistency:
    """Test consistency between forward and generate."""

    def test_first_token_matches_argmax(self, model):
        """First generated token should match argmax of last-token logits."""
        torch.manual_seed(123)
        model.model.eval()

        image = generate_random_image(seed=123)
        inputs = model.prepare_messages("Describe the image.", image)

        # Get logits from forward pass
        with torch.no_grad():
            forward_outputs = model.forward(inputs)

        # Last token's logits predict the first generated token
        last_token_logits = forward_outputs.logits[0, -1, :]
        predicted_first_token = last_token_logits.argmax().item()

        # Generate with greedy decoding
        with torch.no_grad():
            gen_outputs = model.generate(inputs, max_new_tokens=1, do_sample=False)

        # Extract the first newly generated token
        input_len = inputs["input_ids"].shape[1]
        actual_first_token = gen_outputs.sequences[0, input_len].item()

        assert predicted_first_token == actual_first_token, (
            f"Forward argmax {predicted_first_token} != generated {actual_first_token}"
        )
