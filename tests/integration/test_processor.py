"""Integration tests for processor/tokenizer behavior.

These tests verify that the processor correctly handles image+text inputs
using SmolVLM-256M as the test model.
"""

import pytest
from PIL import Image
import numpy as np


def generate_test_image(width=56, height=56, seed=None):
    """Generate a simple test image."""
    if seed is not None:
        np.random.seed(seed)
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestProcessorImageTokens:
    """Test that processor correctly handles image tokens."""

    def test_processor_adds_image_tokens(self, tiny_model):
        """Processed input should contain image placeholder tokens."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe this image.", image)

        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = tiny_model.adapter.get_image_token_id()

        # Should contain at least one image token
        image_token_count = (input_ids == image_token_id).sum().item()
        assert image_token_count > 0, "No image tokens in processed input"

    def test_processor_handles_multiple_images(self, tiny_model):
        """Processor should handle multiple images (if supported)."""
        images = [generate_test_image(seed=i) for i in range(2)]
        tasks = ["Describe image 1.", "Describe image 2."]

        # Test batch preparation
        inputs = tiny_model.prepare_messages_batch(tasks, images)

        assert inputs["input_ids"].shape[0] == 2  # Batch size 2
        assert "attention_mask" in inputs

    def test_processed_input_has_required_keys(self, tiny_model):
        """Processed input should have input_ids and attention_mask."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape == inputs["attention_mask"].shape

    def test_input_ids_shape(self, tiny_model):
        """input_ids should be 2D tensor [batch=1, seq_len]."""
        image = generate_test_image(seed=1)
        inputs = tiny_model.prepare_messages("Describe.", image)

        assert len(inputs["input_ids"].shape) == 2
        assert inputs["input_ids"].shape[0] == 1  # Batch size


class TestProcessorBatch:
    """Test batch processing behavior."""

    def test_batch_sizes_match(self, tiny_model):
        """Batch inputs should have consistent sizes."""
        images = [generate_test_image(seed=i) for i in range(3)]
        tasks = ["Task 1", "Task 2", "Task 3"]

        inputs = tiny_model.prepare_messages_batch(tasks, images)

        assert inputs["input_ids"].shape[0] == 3
        assert inputs["attention_mask"].shape[0] == 3

    def test_batch_mismatched_raises(self, tiny_model):
        """Mismatched task/image counts should raise ValueError."""
        images = [generate_test_image(seed=1)]
        tasks = ["Task 1", "Task 2"]  # More tasks than images

        with pytest.raises(ValueError, match="must match"):
            tiny_model.prepare_messages_batch(tasks, images)

    def test_batch_return_text(self, tiny_model):
        """prepare_messages_batch with return_text should return texts."""
        images = [generate_test_image(seed=1)]
        tasks = ["Describe."]

        inputs, texts = tiny_model.prepare_messages_batch(tasks, images, return_text=True)

        assert isinstance(texts, list)
        assert len(texts) == 1
        assert isinstance(texts[0], str)


class TestProcessorTokenizer:
    """Test tokenizer behavior via processor."""

    def test_tokenizer_accessible(self, tiny_model):
        """Tokenizer should be accessible through processor."""
        tokenizer = tiny_model.processor.tokenizer
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    def test_vocab_size_positive(self, tiny_model):
        """Tokenizer vocab size should be positive."""
        vocab_size = len(tiny_model.processor.tokenizer)
        assert vocab_size > 0

    def test_special_tokens_exist(self, tiny_model):
        """Tokenizer should have standard special tokens."""
        tokenizer = tiny_model.processor.tokenizer

        # Most tokenizers have these
        assert tokenizer.pad_token_id is not None or tokenizer.eos_token_id is not None
