"""Tests for forward passes and generation - Concern #1.

These tests validate that a model implementation correctly produces:
- Valid logits from forward passes
- Valid token sequences from generation
- Proper hidden state and attention extraction
- Correct gradient computation
"""

import torch
import pytest

from conftest import generate_random_image


class TestForward:
    """Test forward pass functionality."""

    def test_forward_returns_logits(self, model):
        """Forward pass should return logits tensor."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs)

        assert hasattr(outputs, "logits")
        assert isinstance(outputs.logits, torch.Tensor)

    def test_forward_logits_shape_matches_vocab(self, model):
        """Logits shape should be [batch, seq_len, vocab_size]."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs)

        assert len(outputs.logits.shape) == 3
        assert outputs.logits.shape[0] == 1  # batch size
        assert outputs.logits.shape[1] > 0  # sequence length
        assert outputs.logits.shape[2] > 0  # vocab size

    def test_forward_logits_vocab_size(self, model):
        """Logits vocab dimension should match tokenizer size."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs)

        tokenizer_vocab_size = len(model.processor.tokenizer)
        logits_vocab_size = outputs.logits.shape[2]

        # Logits vocab should match tokenizer (including special tokens)
        assert logits_vocab_size >= tokenizer_vocab_size, (
            f"Logits vocab {logits_vocab_size} < tokenizer vocab {tokenizer_vocab_size}"
        )

    def test_forward_with_hidden_states(self, model):
        """Forward with output_hidden_states=True should return hidden states."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs, output_hidden_states=True)

        assert hasattr(outputs, "hidden_states")
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) > 0
        # Each hidden state should be a tensor
        assert isinstance(outputs.hidden_states[0], torch.Tensor)

    def test_forward_with_attentions(self, model):
        """Forward with output_attentions=True should return attention weights."""
        # Use smaller image - attention matrices are O(seq_len^2) memory
        image = generate_random_image(width=200, height=200)
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs, output_attentions=True)

        assert hasattr(outputs, "attentions")
        assert isinstance(outputs.attentions, tuple)
        assert len(outputs.attentions) > 0
        assert isinstance(outputs.attentions[0], torch.Tensor)

    def test_forward_with_gradients(self, model):
        """Forward with require_grads=True should allow gradient computation."""
        image = generate_random_image(width=28, height=28)  # smaller for memory
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs, require_grads=True)

        # Should be able to compute gradients
        loss = outputs.logits[0, 0, :10].sum()
        loss.backward()

    def test_forward_no_nan_or_inf(self, model):
        """Forward pass should not produce NaN or Inf values."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.forward(inputs)

        assert not torch.isnan(outputs.logits).any(), "Logits contain NaN values"
        assert not torch.isinf(outputs.logits).any(), "Logits contain Inf values"


class TestGenerate:
    """Test generation functionality."""

    def test_generate_returns_sequences(self, model):
        """Generate should return token sequences."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.generate(inputs, max_new_tokens=10)

        assert hasattr(outputs, "sequences")
        assert isinstance(outputs.sequences, torch.Tensor)

    def test_generate_valid_token_ids(self, model):
        """Generated token IDs should be within valid range."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.generate(inputs, max_new_tokens=10)

        # Use len(tokenizer) to include added special tokens
        max_token_id = len(model.processor.tokenizer)
        sequences = outputs.sequences

        assert (sequences >= 0).all(), "Generated negative token IDs"
        assert (sequences < max_token_id).all(), (
            f"Generated token ID >= {max_token_id}"
        )

    def test_generate_sequence_length(self, model):
        """Generated sequence length should be reasonable."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        input_len = inputs["input_ids"].shape[1]
        max_new_tokens = 10

        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
        output_len = outputs.sequences.shape[1]

        # Output should be non-empty and longer than input
        assert output_len > 0, "Generated empty sequence"
        assert output_len >= input_len, "Output shorter than input"
        # Should not exceed input + max_new_tokens
        assert output_len <= input_len + max_new_tokens, (
            f"Generated {output_len - input_len} tokens, expected <= {max_new_tokens}"
        )

    def test_generate_with_hidden_states(self, model):
        """Generate with output_hidden_states=True should return hidden states."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.generate(inputs, max_new_tokens=5, output_hidden_states=True)

        assert hasattr(outputs, "hidden_states")
        assert isinstance(outputs.hidden_states, tuple)
        assert len(outputs.hidden_states) > 0
        # Hidden states is a tuple per generated step
        assert isinstance(outputs.hidden_states[0], tuple)
        assert len(outputs.hidden_states[0]) > 0
        assert isinstance(outputs.hidden_states[0][0], torch.Tensor)

    def test_generate_deterministic_without_sampling(self, model):
        """Generation with do_sample=False should be deterministic."""
        # Seed everything for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model.model.eval()

        image = generate_random_image(seed=42)
        inputs = model.prepare_messages("Describe the image.", image)

        outputs1 = model.generate(inputs, max_new_tokens=10, do_sample=False)

        # Re-seed and run again
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        outputs2 = model.generate(inputs, max_new_tokens=10, do_sample=False)

        tokenizer = model.processor.tokenizer
        text1 = tokenizer.decode(outputs1.sequences[0], skip_special_tokens=True)
        text2 = tokenizer.decode(outputs2.sequences[0], skip_special_tokens=True)

        assert text1 == text2, f"Expected deterministic output, got '{text1}' vs '{text2}'"

    def test_generate_with_gradients(self, model):
        """Generate with require_grads=True should work."""
        image = generate_random_image(width=28, height=28)
        inputs = model.prepare_messages("Describe the image.", image)
        outputs = model.generate(inputs, max_new_tokens=2, require_grads=True)

        assert hasattr(outputs, "sequences")

    def test_generate_with_prefill(self, model):
        """Generate with assistant prefill should include prefill in output."""
        image = generate_random_image()
        prefill = '{"description": "'

        inputs = model.prepare_messages(
            "Format your response as JSON.",
            image,
            assistant_prefill=prefill,
        )
        outputs = model.generate(inputs, max_new_tokens=50)

        generated_text = model.processor.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        assert prefill in generated_text


class TestForwardGenerateConsistency:
    """Test consistency between forward and generate."""

    def test_first_token_matches_argmax(self, model):
        """First generated token should match argmax of last-token logits.

        With do_sample=False, greedy decoding should select the token with
        highest logit probability at each step.
        """
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
