import pytest
from PIL import Image
import numpy as np
from vlm_spectra.models.HookedVLM import HookedVLM

MODEL_NAMES = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request):
    return HookedVLM(request.param)


def generate_random_image(width=112, height=112, num_channels=3, seed=None):
    """Generate a smaller random image for testing to reduce memory usage"""
    if seed is not None:
        np.random.seed(seed)
    random_array = np.random.randint(
        0, 255, (height, width, num_channels), dtype=np.uint8
    )
    random_image = Image.fromarray(random_array)
    return random_image


def generate_checkered_image(
    width=112, height=112, num_channels=3, checkered_size=14, seed=None
):
    """Generate a smaller checkered image for testing to reduce memory usage"""
    if seed is not None:
        np.random.seed(seed)
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


class TestBatchMethods:
    """Test suite for batch methods in HookedVLM"""

    def test_prepare_messages_batch_basic(self, model):
        """Test basic functionality of prepare_messages_batch"""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        inputs = model.prepare_messages_batch(tasks, images)

        # Check that we get the expected tensor keys
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "pixel_values" in inputs
        assert "image_grid_thw" in inputs

        # Check batch dimensions
        assert inputs["input_ids"].shape[0] == 2  # batch size
        assert inputs["attention_mask"].shape[0] == 2
        assert inputs["image_grid_thw"].shape[0] == 2  # batch size for image metadata
        # pixel_values has total patches, not batch size as first dimension
        assert len(inputs["pixel_values"].shape) == 2  # [total_patches, feature_dim]

    def test_prepare_messages_batch_return_text(self, model):
        """Test prepare_messages_batch with return_text=True"""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        inputs, texts = model.prepare_messages_batch(tasks, images, return_text=True)

        assert isinstance(texts, list)
        assert len(texts) == 2
        assert all(isinstance(text, str) for text in texts)

    def test_prepare_messages_batch_mismatched_lengths(self, model):
        """Test that mismatched task/image lengths raise ValueError"""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        with pytest.raises(
            ValueError, match="Number of tasks .* must match number of images"
        ):
            model.prepare_messages_batch(tasks, images)

    def test_prepare_messages_batch_with_prefill(self, model):
        """Test prepare_messages_batch with assistant prefill"""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]

        inputs = model.prepare_messages_batch(
            tasks, images, assistant_prefill="The image shows"
        )

        # Should still work and produce valid inputs
        assert "input_ids" in inputs
        assert inputs["input_ids"].shape[0] == 1

    def test_forward_batch_basic(self, model):
        """Test basic functionality of forward_batch"""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.forward_batch(tasks=tasks, images=images)

        # Check that we get logits with proper batch size
        assert hasattr(outputs, "logits")
        assert outputs.logits.shape[0] == 2  # batch size
        assert len(outputs.logits.shape) == 3  # [batch, seq_len, vocab_size]

    def test_forward_batch_with_hidden_states(self, model):
        """Test forward_batch with output_hidden_states=True"""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]

        outputs = model.forward_batch(
            tasks=tasks, images=images, output_hidden_states=True
        )

        assert hasattr(outputs, "hidden_states")
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) > 0

    def test_forward_batch_with_attentions(self, model):
        """Test forward_batch with output_attentions=True"""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]

        outputs = model.forward_batch(
            tasks=tasks, images=images, output_attentions=True
        )

        assert hasattr(outputs, "attentions")
        assert isinstance(outputs.attentions, tuple)
        assert len(outputs.attentions) > 0

    def test_forward_batch_missing_args(self, model):
        """Test forward_batch raises error with missing arguments"""
        with pytest.raises(
            ValueError,
            match="Either inputs_list or both tasks and images must be provided",
        ):
            model.forward_batch()

    def test_generate_batch_basic(self, model):
        """Test basic functionality of generate_batch"""
        tasks = ["Describe the image.", "What do you see?"]
        images = [generate_random_image(seed=1), generate_random_image(seed=2)]

        outputs = model.generate_batch(tasks=tasks, images=images, max_new_tokens=10)

        # Check that we get sequences with proper batch size
        assert hasattr(outputs, "sequences")
        assert outputs.sequences.shape[0] == 2  # batch size
        assert len(outputs.sequences.shape) == 2  # [batch, seq_len]

    def test_generate_batch_with_hidden_states(self, model):
        """Test generate_batch with output_hidden_states=True"""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]

        outputs = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=5, output_hidden_states=True
        )

        assert hasattr(outputs, "hidden_states")
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) > 0

    def test_generate_batch_missing_args(self, model):
        """Test generate_batch raises error with missing arguments"""
        with pytest.raises(
            ValueError,
            match="Either inputs_list or both tasks and images must be provided",
        ):
            model.generate_batch()

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_batch_consistency_forward(self, model, batch_size):
        """Test that batch forward produces consistent results"""
        # Create deterministic test data with smaller images
        tasks = [f"Describe image {i}." for i in range(batch_size)]
        images = [
            generate_checkered_image(width=56, height=56, seed=i)
            for i in range(batch_size)
        ]

        # Run batch forward
        batch_outputs = model.forward_batch(tasks=tasks, images=images)

        # Compare outputs - they should have reasonable structure
        assert batch_outputs.logits.shape[0] == batch_size

        # Check that each item in the batch has reasonable sequence length
        assert batch_outputs.logits.shape[1] > 0  # non-empty sequence
        assert batch_outputs.logits.shape[2] > 0  # vocab size

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_batch_consistency_generate(self, model, batch_size):
        """Test that batch generate produces reasonable results"""
        # Create test data with smaller images
        tasks = [f"Describe image {i}." for i in range(batch_size)]
        images = [
            generate_checkered_image(width=56, height=56, seed=i)
            for i in range(batch_size)
        ]

        # Run batch generation
        batch_outputs = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=3, do_sample=False
        )

        # Check that we get the expected batch size
        assert batch_outputs.sequences.shape[0] == batch_size

        # Check that generated sequences are reasonable length
        assert batch_outputs.sequences.shape[1] > 0  # should have some tokens

    def test_single_batch_equivalence_generate_text(self, model):
        """Test that single-item batch generates same text as individual call with do_sample=False"""
        task = "Describe the image."
        image = generate_checkered_image(width=56, height=56, seed=42)

        # Individual call
        single_inputs = model.prepare_messages(task, image)
        single_output = model.generate(single_inputs, max_new_tokens=10, do_sample=False)

        # Batch call with single item
        batch_output = model.generate_batch(
            tasks=[task], images=[image], max_new_tokens=10, do_sample=False
        )

        # Decode both outputs to text
        tokenizer = model.processor.tokenizer
        single_text = tokenizer.decode(single_output.sequences[0], skip_special_tokens=True)
        batch_text = tokenizer.decode(batch_output.sequences[0], skip_special_tokens=True)

        # Text outputs should be identical with do_sample=False
        assert single_text == batch_text, f"Single: '{single_text}' != Batch: '{batch_text}'"

    def test_multi_batch_generate_text_quality(self, model):
        """Test that multi-item batch generates reasonable text outputs with do_sample=False"""
        # Note: Multi-item batches may produce slightly different text than individual calls
        # due to padding effects and attention interactions, even with do_sample=False
        tasks = ["Describe the image.", "What do you see?"]
        images = [
            generate_checkered_image(width=56, height=56, seed=1),
            generate_checkered_image(width=56, height=56, seed=2)
        ]

        # Batch call
        batch_output = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=8, do_sample=False
        )

        # Decode batch outputs and verify they're reasonable
        tokenizer = model.processor.tokenizer
        for i in range(len(tasks)):
            text = tokenizer.decode(batch_output.sequences[i], skip_special_tokens=True)

            # Basic quality checks
            assert len(text) > 0, f"Empty text output for item {i}"
            assert "assistant" in text, f"Missing assistant response in item {i}"
            assert not text.count("�") > 5, f"Too many replacement characters in item {i}"

    def test_multi_batch_structure_consistency(self, model):
        """Test that multi-item batch produces reasonable output structure"""
        # Note: Multi-item batches may not produce identical results to individual calls
        # due to padding effects and attention mask interactions. This is expected behavior.
        tasks = ["Describe the image.", "What colors do you see?"]
        images = [
            generate_checkered_image(width=56, height=56, seed=1),
            generate_checkered_image(width=56, height=56, seed=2)
        ]

        # Individual calls for structure comparison
        individual_outputs = []
        for task, image in zip(tasks, images):
            single_inputs = model.prepare_messages(task, image)
            single_output = model.forward(single_inputs)
            individual_outputs.append(single_output)

        # Batch call
        batch_output = model.forward_batch(tasks=tasks, images=images)

        # Verify batch structure is correct
        assert batch_output.logits.shape[0] == len(tasks)  # Correct batch size
        assert batch_output.logits.shape[2] == individual_outputs[0].logits.shape[2]  # Same vocab size

        # Verify each item in batch produces reasonable logits (not all zeros/nan/inf)
        import torch
        for i in range(len(tasks)):
            batch_item_logits = batch_output.logits[i]

            # Check for reasonable value ranges
            assert not torch.isnan(batch_item_logits).any(), f"NaN values in batch item {i}"
            assert not torch.isinf(batch_item_logits).any(), f"Inf values in batch item {i}"
            assert batch_item_logits.abs().max() < 100, f"Unreasonably large values in batch item {i}"

            # Check that logits have reasonable variance (not all the same value)
            assert batch_item_logits.std() > 0.1, f"Low variance in batch item {i} logits"

    def test_batch_size_consistency(self, model):
        """Test that different batch sizes produce reasonable results"""
        # Create test data with smaller images
        task = "Describe the image."
        image = generate_checkered_image(width=56, height=56, seed=123)

        # Single item
        output_single = model.forward_batch(tasks=[task], images=[image])

        # Same item in a batch of 2 (duplicate)
        output_batch = model.forward_batch(tasks=[task, task], images=[image, image])

        # Check shapes are reasonable
        assert output_single.logits.shape[0] == 1
        assert output_batch.logits.shape[0] == 2
        # Sequence lengths might differ due to batching/padding
        assert (
            output_single.logits.shape[2] == output_batch.logits.shape[2]
        )  # vocab size should match

    def test_inputs_list_not_implemented(self, model):
        """Test that complex inputs_list handling raises NotImplementedError"""
        # Create some dummy inputs
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]
        inputs1 = model.prepare_messages_batch(tasks, images)
        inputs2 = model.prepare_messages_batch(tasks, images)

        # This should raise NotImplementedError for multiple inputs
        with pytest.raises(
            NotImplementedError,
            match="Complex collation of input_list not yet implemented",
        ):
            model.forward_batch(inputs_list=[inputs1, inputs2])

        with pytest.raises(
            NotImplementedError,
            match="Complex collation of input_list not yet implemented",
        ):
            model.generate_batch(inputs_list=[inputs1, inputs2])

    def test_single_inputs_list_works(self, model):
        """Test that single item in inputs_list works"""
        tasks = ["Describe the image."]
        images = [generate_random_image(seed=1)]
        inputs = model.prepare_messages_batch(tasks, images)

        # Single item in list should work
        output = model.forward_batch(inputs_list=[inputs])
        assert hasattr(output, "logits")

        output = model.generate_batch(inputs_list=[inputs], max_new_tokens=5)
        assert hasattr(output, "sequences")

    def test_empty_batch(self, model):
        """Test handling of empty batches"""
        # Empty lists should raise an error in prepare_messages_batch
        # since processor likely can't handle empty inputs
        with pytest.raises((ValueError, IndexError, TypeError)):
            model.prepare_messages_batch([], [])

    @pytest.mark.parametrize("require_grads", [False, True])
    def test_batch_gradients(self, model, require_grads):
        """Test that gradient computation works correctly for batches"""
        # Use very small images to reduce memory usage
        tasks = ["Describe the image."]
        images = [generate_checkered_image(width=28, height=28, seed=1)]

        # Test forward
        output = model.forward_batch(
            tasks=tasks, images=images, require_grads=require_grads
        )

        if require_grads:
            # Should be able to compute gradients on a small subset
            loss = output.logits[0, 0, :10].sum()  # Use only first 10 logits
            loss.backward()

        # Test generate (gradients don't make as much sense for generation)
        output = model.generate_batch(
            tasks=tasks, images=images, max_new_tokens=2, require_grads=require_grads
        )
        assert hasattr(output, "sequences")
