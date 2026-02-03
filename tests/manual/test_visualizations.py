"""Manual verification tests for visualizations.

These tests generate visual outputs that require human inspection.
Run with: uv run pytest tests/manual/test_visualizations.py --run-manual -v
"""

import pytest
from PIL import Image
import numpy as np

from vlm_spectra import HookedVLM


def generate_checkered_image(width=224, height=224, checkered_size=28):
    """Generate a checkered pattern image."""
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


@pytest.fixture(scope="module")
def model_for_manual():
    """Load smallest model for manual tests."""
    return HookedVLM.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")


@pytest.mark.manual
def test_patch_overview_image(model_for_manual, tmp_path):
    """
    Manual verification for patch overview visualization.

    Instructions:
    1. Run: uv run pytest tests/manual/test_visualizations.py::test_patch_overview_image --run-manual -v
    2. Open the printed image path
    3. Verify:
       - [ ] Grid lines are visible (red)
       - [ ] Patch numbers are readable
       - [ ] Patches align with image content
    """
    image = generate_checkered_image(width=224, height=224)
    result = model_for_manual.generate_patch_overview(image, with_labels=True)

    output_path = tmp_path / "patch_overview_test.png"
    result.save(output_path)

    print(f"\n\n=== MANUAL VERIFICATION REQUIRED ===")
    print(f"Open this file to verify: {output_path}")
    print(f"Checklist:")
    print(f"  - [ ] Grid lines are visible (red)")
    print(f"  - [ ] Patch numbers are readable (1, 11, 21, etc.)")
    print(f"  - [ ] Patches align with checkered pattern")
    print(f"=====================================\n")


@pytest.mark.manual
def test_patch_overview_no_labels(model_for_manual, tmp_path):
    """
    Manual verification for patch overview without labels.

    Instructions:
    1. Run: uv run pytest tests/manual/test_visualizations.py::test_patch_overview_no_labels --run-manual -v
    2. Open the printed image path
    3. Verify:
       - [ ] Grid lines are visible (red)
       - [ ] No labels/numbers appear
       - [ ] Image content is clearly visible through grid
    """
    image = generate_checkered_image(width=224, height=224)
    result = model_for_manual.generate_patch_overview(image, with_labels=False)

    output_path = tmp_path / "patch_overview_no_labels.png"
    result.save(output_path)

    print(f"\n\n=== MANUAL VERIFICATION REQUIRED ===")
    print(f"Open this file to verify: {output_path}")
    print(f"Checklist:")
    print(f"  - [ ] Grid lines visible but no labels")
    print(f"  - [ ] Checkered pattern clearly visible")
    print(f"=====================================\n")


@pytest.mark.manual
def test_model_basic_generation(model_for_manual, tmp_path):
    """
    Manual verification for basic model generation.

    Instructions:
    1. Run: uv run pytest tests/manual/test_visualizations.py::test_model_basic_generation --run-manual -v
    2. Review the printed output
    3. Verify:
       - [ ] Generated text is coherent
       - [ ] Text relates to the checkered pattern
       - [ ] No obvious errors or gibberish
    """
    image = generate_checkered_image(width=224, height=224)
    inputs = model_for_manual.prepare_messages("Describe this image.", image)

    outputs = model_for_manual.generate(inputs, max_new_tokens=50, do_sample=False)

    tokenizer = model_for_manual.processor.tokenizer
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    print(f"\n\n=== MANUAL VERIFICATION REQUIRED ===")
    print(f"Generated text:")
    print(f"---")
    print(generated_text)
    print(f"---")
    print(f"Checklist:")
    print(f"  - [ ] Text is coherent (not gibberish)")
    print(f"  - [ ] Text relates to checkered/pattern image")
    print(f"=====================================\n")


@pytest.mark.manual
def test_cache_activation_sample(model_for_manual, tmp_path):
    """
    Manual verification for cached activations.

    Instructions:
    1. Run: uv run pytest tests/manual/test_visualizations.py::test_cache_activation_sample --run-manual -v
    2. Review the printed statistics
    3. Verify:
       - [ ] Activation values are reasonable (not NaN/Inf)
       - [ ] Mean/std values are in expected range
       - [ ] Shape matches expected dimensions
    """
    image = generate_checkered_image(width=56, height=56)
    inputs = model_for_manual.prepare_messages("Describe.", image)

    with model_for_manual.run_with_cache(["lm.blocks.*.hook_resid_post"]):
        model_for_manual.forward(inputs)

    # Get layer 0 activations
    layer0 = model_for_manual.cache["lm.blocks.0.hook_resid_post"]

    print(f"\n\n=== MANUAL VERIFICATION REQUIRED ===")
    print(f"Layer 0 residual post activations:")
    print(f"  Shape: {layer0.shape}")
    print(f"  Mean: {layer0.float().mean().item():.4f}")
    print(f"  Std: {layer0.float().std().item():.4f}")
    print(f"  Min: {layer0.float().min().item():.4f}")
    print(f"  Max: {layer0.float().max().item():.4f}")
    print(f"  Contains NaN: {layer0.isnan().any().item()}")
    print(f"  Contains Inf: {layer0.isinf().any().item()}")
    print(f"Checklist:")
    print(f"  - [ ] No NaN or Inf values")
    print(f"  - [ ] Mean/std are reasonable (not extremely large/small)")
    print(f"  - [ ] Shape has 3 dimensions [batch, seq_len, hidden_dim]")
    print(f"=====================================\n")


@pytest.mark.manual
def test_photo_image_patch_overview(model_for_manual, tmp_path):
    """
    Manual verification for patch overview with a photo-like image.

    Instructions:
    1. Run: uv run pytest tests/manual/test_visualizations.py::test_photo_image_patch_overview --run-manual -v
    2. Open the printed image path
    3. Verify:
       - [ ] Grid overlay is visible on the gradient
       - [ ] Labels are readable
       - [ ] Gradient colors preserved through overlay
    """
    # Create a gradient image (simulating a photo)
    width, height = 224, 224
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            arr[y, x] = [
                int(255 * x / width),  # Red gradient left to right
                int(255 * y / height),  # Green gradient top to bottom
                128,  # Constant blue
            ]
    image = Image.fromarray(arr)

    result = model_for_manual.generate_patch_overview(image, with_labels=True)

    output_path = tmp_path / "gradient_patch_overview.png"
    result.save(output_path)

    print(f"\n\n=== MANUAL VERIFICATION REQUIRED ===")
    print(f"Open this file to verify: {output_path}")
    print(f"Checklist:")
    print(f"  - [ ] Red-to-green gradient visible")
    print(f"  - [ ] Grid lines clearly overlay on gradient")
    print(f"  - [ ] Patch numbers readable on all backgrounds")
    print(f"=====================================\n")
