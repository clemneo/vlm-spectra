"""Core contract tests for HookedVLM.

These are the minimal tests that MUST pass for any supported model.
They define the essential API guarantees.
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


class ZeroingHook:
    """Hook that zeros out layer output."""

    hook_point = "lm.layer.post"

    def __init__(self, layer):
        self.layer = layer

    def __call__(self, module, args, kwargs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


class TestCoreContracts:
    """Minimal tests that MUST pass for any supported model."""

    # --- Contract 1: Basic Forward ---

    def test_forward_returns_valid_logits(self, model):
        """Contract 1: forward() returns [batch, seq, vocab] logits."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)
        outputs = model.forward(inputs)

        assert hasattr(outputs, "logits")
        assert len(outputs.logits.shape) == 3  # [batch=1, seq_len, vocab_size]
        assert not torch.isnan(outputs.logits).any()

    # --- Contract 2: Basic Generate ---

    def test_generate_returns_valid_tokens(self, model):
        """Contract 2: generate() returns valid token sequences."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)
        outputs = model.generate(inputs, max_new_tokens=5)

        assert hasattr(outputs, "sequences")
        assert (outputs.sequences >= 0).all()
        assert (outputs.sequences < len(model.processor.tokenizer)).all()

    # --- Contract 3: Deterministic Generation ---

    def test_generate_deterministic_without_sampling(self, model):
        """Contract 3: do_sample=False produces identical outputs."""
        model.model.eval()
        image = generate_random_image(seed=42)
        inputs = model.prepare_messages("Describe.", image)

        out1 = model.generate(inputs, max_new_tokens=5, do_sample=False)
        out2 = model.generate(inputs, max_new_tokens=5, do_sample=False)

        assert torch.equal(out1.sequences, out2.sequences)

    # --- Contract 4: Cache Captures Activations ---

    def test_cache_captures_activations(self, model):
        """Contract 4: run_with_cache captures tensors at hook points."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)

        with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            model.forward(inputs)

        assert model.cache is not None
        assert "lm.blocks.0.hook_resid_post" in model.cache
        assert isinstance(model.cache["lm.blocks.0.hook_resid_post"], torch.Tensor)

    # --- Contract 5: Hooks Modify Output ---

    def test_hooks_modify_output(self, model):
        """Contract 5: run_with_hooks can modify layer outputs."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)

        original = model.forward(inputs).logits.clone()

        hook = ZeroingHook(layer=0)
        with model.run_with_hooks([hook]):
            modified = model.forward(inputs).logits

        assert not torch.allclose(original, modified, atol=1e-3)

    # --- Contract 6: Consistent Hidden State Shapes ---

    def test_hidden_state_shapes_consistent(self, model):
        """Contract 6: All layers produce same-shaped hidden states."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)

        with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
            model.forward(inputs)

        # Each layer's hidden state: [batch=1, seq_len, hidden_dim]
        shapes = [
            model.cache[f"lm.blocks.{i}.hook_resid_post"].shape
            for i in range(model.lm_num_layers)
        ]
        assert all(s == shapes[0] for s in shapes)

    # --- Contract 7: Gradient Support ---

    def test_forward_supports_gradients(self, model):
        """Contract 7: forward(require_grads=True) allows backward()."""
        image = generate_random_image(width=28, height=28)
        inputs = model.prepare_messages("Describe.", image)
        outputs = model.forward(inputs, require_grads=True)

        loss = outputs.logits[0, -1, :10].sum()
        loss.backward()  # Should not raise

        # Cleanup
        model.model.zero_grad(set_to_none=True)
        model.model.requires_grad_(False)

    # --- Contract 8: Component Access ---

    def test_components_accessible(self, model):
        """Contract 8: get_model_components returns norm, lm_head, tokenizer."""
        components = model.get_model_components()

        assert "norm" in components
        assert "lm_head" in components
        assert "tokenizer" in components

    # --- Contract 9: Image Token Range ---

    def test_image_token_range_valid(self, model):
        """Contract 9: get_image_token_range returns valid indices."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)

        start, end = model.get_image_token_range(inputs)

        assert isinstance(start, int)
        assert isinstance(end, int)
        assert 0 <= start <= end < inputs["input_ids"].shape[1]

    # --- Contract 10: Input Preparation ---

    def test_prepare_messages_returns_valid_inputs(self, model):
        """Contract 10: prepare_messages returns dict with required keys."""
        image = generate_random_image()
        inputs = model.prepare_messages("Describe.", image)

        # Must have these keys
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

        # input_ids: [batch=1, seq_len] - 2D tensor of token IDs
        assert len(inputs["input_ids"].shape) == 2
        assert inputs["input_ids"].shape[0] == 1

        # attention_mask: same shape as input_ids
        assert inputs["attention_mask"].shape == inputs["input_ids"].shape
