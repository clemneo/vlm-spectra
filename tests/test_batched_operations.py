"""Tests for batched forward passes and generation - Concern #2.

These tests validate that a model implementation correctly handles:
- Batched input preparation
- Batched forward passes
- Batched generation
- Consistency between single and batched operations
"""

import torch
import pytest

from conftest import generate_random_image, generate_checkered_image


class TestBatchedForward:
    """Test batched forward pass functionality."""

    def test_batch_forward_returns_logits(self, model):
        """Batch forward should return logits for all items."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.forward_batch(tasks=tasks, images=images)

        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 2  # batch size

    def test_batch_forward_shape_consistency(self, model):
        """Batch forward should produce consistent shapes."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.forward_batch(tasks=tasks, images=images)

        assert len(outputs.logits.shape) == 3  # [batch, seq_len, vocab_size]
        assert outputs.logits.shape[0] == 2  # batch size
        assert outputs.logits.shape[1] > 0  # sequence length
        assert outputs.logits.shape[2] > 0  # vocab size

    def test_batch_single_item_equals_unbatched(self, model):
        """Single-item batch should produce same logits as unbatched call."""
        model.model.eval()
        task = "Describe the image."
        image = generate_checkered_image(width=56, height=56, seed=42)

        # Single call
        single_inputs = model.prepare_messages(task, image)
        single_output = model.forward(single_inputs)

        # Batch call with single item
        batch_output = model.forward_batch(tasks=[task], images=[image])

        # The logits should match (allowing for padding differences)
        single_seq_len = single_output.logits.shape[1]
        batch_logits_trimmed = batch_output.logits[0, :single_seq_len, :]

        assert torch.allclose(
            single_output.logits[0],
            batch_logits_trimmed,
            atol=5e-3,
        ), "Single and batched forward should produce matching logits"

    def test_batch_multi_item_no_nan_or_inf(self, model):
        """Multi-item batch should not produce NaN or Inf values."""
        tasks = ["Describe the image.", "What colors do you see?"]
        images = [
            generate_checkered_image(width=56, height=56, seed=1),
            generate_checkered_image(width=56, height=56, seed=2),
        ]

        outputs = model.forward_batch(tasks=tasks, images=images)

        for i in range(len(tasks)):
            assert not torch.isnan(outputs.logits[i]).any(), f"NaN in batch item {i}"
            assert not torch.isinf(outputs.logits[i]).any(), f"Inf in batch item {i}"

    def test_batch_with_hidden_states(self, model):
        """Batch forward with output_hidden_states=True should work."""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]

        outputs = model.forward_batch(
            tasks=tasks, images=images, output_hidden_states=True
        )

        assert hasattr(outputs, "hidden_states")
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) > 0


class TestBatchedGenerate:
    """Test batched generation functionality."""

    def test_batch_generate_returns_sequences(self, model):
        """Batch generate should return sequences for all items."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.generate_batch(tasks=tasks, images=images, max_new_tokens=10)

        assert hasattr(outputs, "sequences")
        assert outputs.sequences.shape[0] == 2  # batch size
        assert len(outputs.sequences.shape) == 2  # [batch, seq_len]
        assert outputs.sequences.shape[1] > 0  # non-empty sequences

    def test_batch_generate_deterministic(self, model):
        """Batch generate with do_sample=False should be deterministic."""
        model.model.eval()
        torch.manual_seed(0)
        tasks = ["Describe the image."]
        images = [generate_checkered_image(width=56, height=56, seed=42)]

        outputs1 = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=5, do_sample=False
        )
        outputs2 = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=5, do_sample=False
        )

        tokenizer = model.processor.tokenizer
        text1 = tokenizer.decode(outputs1.sequences[0], skip_special_tokens=True)
        text2 = tokenizer.decode(outputs2.sequences[0], skip_special_tokens=True)

        assert text1 == text2

    def test_batch_single_item_equals_unbatched(self, model):
        """Single-item batch should produce same text as unbatched call."""
        model.model.eval()
        torch.manual_seed(0)
        task = "Describe the image."
        image = generate_checkered_image(width=56, height=56, seed=42)

        # Single call
        single_inputs = model.prepare_messages(task, image)
        single_output = model.generate(single_inputs, max_new_tokens=10, do_sample=False)

        # Batch call with single item
        batch_output = model.generate_batch(
            tasks=[task], images=[image], max_new_tokens=10, do_sample=False
        )

        tokenizer = model.processor.tokenizer
        single_text = tokenizer.decode(single_output.sequences[0], skip_special_tokens=True)
        batch_text = tokenizer.decode(batch_output.sequences[0], skip_special_tokens=True)

        assert single_text == batch_text, f"'{single_text}' != '{batch_text}'"

    def test_batch_with_hidden_states(self, model):
        """Batch generate with output_hidden_states=True should work."""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]

        outputs = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=5, output_hidden_states=True
        )

        assert hasattr(outputs, "hidden_states")
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) > 0
        assert isinstance(outputs.hidden_states[0], tuple)
        assert len(outputs.hidden_states[0]) > 0
        assert isinstance(outputs.hidden_states[0][0], torch.Tensor)

    def test_batch_generate_sequences_are_valid_tokens(self, model):
        """Generated sequences should contain valid token IDs."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.generate_batch(tasks=tasks, images=images, max_new_tokens=10)

        vocab_size = len(model.processor.tokenizer)
        assert (outputs.sequences >= 0).all()
        assert (outputs.sequences < vocab_size).all()

    def test_batch_generate_respects_max_new_tokens(self, model):
        """Generated sequences should not exceed prompt + max_new_tokens."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        inputs = model.prepare_messages_batch(tasks, images)
        prompt_len = inputs["input_ids"].shape[1]
        max_new_tokens = 8

        outputs = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=max_new_tokens
        )

        assert outputs.sequences.shape[1] <= prompt_len + max_new_tokens


class TestBatchInputPreparation:
    """Test batch input preparation functionality."""

    def test_prepare_messages_batch_returns_tensors(self, model):
        """prepare_messages_batch should return proper tensor structure."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        inputs = model.prepare_messages_batch(tasks, images)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs
        assert "image_grid_thw" in inputs

        assert inputs["input_ids"].shape[0] == 2  # batch size
        assert inputs["attention_mask"].shape[0] == 2

    def test_prepare_messages_batch_mismatched_raises(self, model):
        """Mismatched task/image counts should raise ValueError."""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        with pytest.raises(
            ValueError, match="Number of tasks .* must match number of images"
        ):
            model.prepare_messages_batch(tasks, images)

    def test_prepare_messages_batch_with_return_text(self, model):
        """prepare_messages_batch with return_text=True should return texts."""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        inputs, texts = model.prepare_messages_batch(tasks, images, return_text=True)

        assert isinstance(texts, list)
        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)


class TestBatchErrorHandling:
    """Test batch error handling."""

    def test_forward_batch_missing_args_raises(self, model):
        """forward_batch without proper args should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="Either inputs_list or both tasks and images must be provided",
        ):
            model.forward_batch()

    def test_generate_batch_missing_args_raises(self, model):
        """generate_batch without proper args should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="Either inputs_list or both tasks and images must be provided",
        ):
            model.generate_batch()

    def test_empty_batch_raises(self, model):
        """Empty batches should raise an error."""
        with pytest.raises((ValueError, IndexError, TypeError)):
            model.prepare_messages_batch([], [])

    def test_complex_inputs_list_not_implemented(self, model):
        """Complex inputs_list collation should raise NotImplementedError."""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]
        inputs1 = model.prepare_messages_batch(tasks, images)
        inputs2 = model.prepare_messages_batch(tasks, images)

        with pytest.raises(
            NotImplementedError,
            match="Complex collation of input_list not yet implemented",
        ):
            model.forward_batch(inputs_list=[inputs1, inputs2])


class TestBatchGradients:
    """Test gradient computation in batch operations."""

    @pytest.mark.parametrize("require_grads", [False, True])
    def test_batch_gradients(self, model, require_grads):
        """Gradient computation should work correctly for batches."""
        tasks = ["Describe the image."]
        images = [generate_checkered_image(width=28, height=28, seed=1)]

        output = model.forward_batch(
            tasks=tasks, images=images, require_grads=require_grads
        )

        if require_grads:
            loss = output.logits[0, 0, :10].sum()
            loss.backward()

            # Clean up gradients to free memory for subsequent tests
            model.model.zero_grad(set_to_none=True)
            model.model.requires_grad_(False)
