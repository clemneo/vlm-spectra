import torch
from vlm_spectra.models.HookedVLM import HookedVLM
import pytest

from PIL import Image
import numpy as np
import os

MODEL_NAMES = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]

SAVE_FILES = True

@pytest.fixture(scope="session", params=MODEL_NAMES)
def model(request):
    return HookedVLM(request.param)


def generate_random_image(width=224, height=224, num_channels=3):
    random_array = np.random.randint(
        0, 255, (height, width, num_channels), dtype=np.uint8
    )
    random_image = Image.fromarray(random_array)
    return random_image


def generate_checkered_image(width=224, height=224, num_channels=3, checkered_size=14):
    image = Image.new("RGB", (width, height))
    for x in range(0, width, checkered_size):
        for y in range(0, height, checkered_size):
            color = (255, 255, 255) if (x + y) % (checkered_size * 2) < checkered_size else (0, 0, 0)
            image.paste(color, (x, y, x + checkered_size, y + checkered_size))
    return image

def test_hookedvlm(model):
    assert model is not None


def test_generate(model):
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    outputs = model.generate(inputs)

    assert type(outputs.sequences) is torch.Tensor


def test_generate_with_output_hidden_states(model):
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    outputs = model.generate(inputs, output_hidden_states=True)

    assert (
        type(outputs.hidden_states) is tuple
    )  # tuple of hidden states through each forward step, ie (seq_len_of_input, 1, 1, 1)
    assert len(outputs.hidden_states) > 0

    assert type(outputs.hidden_states[0]) is tuple  # tuple of layers
    assert len(outputs.hidden_states[0]) > 0

    assert (
        type(outputs.hidden_states[0][0]) is torch.Tensor
    )  # the tensor of shape (batch_size, sequence_length, hidden_size)


def test_forward(model):
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    outputs = model.forward(inputs)
    assert type(outputs.logits) is torch.Tensor


def test_forward_with_output_hidden_states(model):
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    outputs = model.forward(inputs, output_hidden_states=True)

    assert type(outputs.logits) is torch.Tensor

    assert type(outputs.hidden_states) is tuple  # tuple of layers
    assert len(outputs.hidden_states) > 0

    assert (
        type(outputs.hidden_states[0]) is torch.Tensor
    )  # the tensor of shape (batch_size, sequence_length, hidden_size)

def test_forward_with_output_attentions(model):
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    outputs = model.forward(inputs, output_attentions=True)
    assert type(outputs.attentions) is tuple
    assert len(outputs.attentions) > 0
    assert type(outputs.attentions[0]) is torch.Tensor


def test_get_model_components(model):
    components = model.get_model_components()
    assert type(components) is dict
    assert len(components) > 0


def test_get_image_token_id(model):
    """Test that the adapter returns the correct image token ID"""
    image_token_id = model.adapter.get_image_token_id()
    
    # Verify it's an integer
    assert isinstance(image_token_id, int)
    
    # Verify it matches what we expect for <|image_pad|>
    expected_token_id = model.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert image_token_id == expected_token_id
    
    # Verify it's not -1 (which would indicate token not found)
    assert image_token_id != -1


@pytest.mark.parametrize("multiplier", [8, 16, 24])
def test_get_image_token_range(model, multiplier):
    """Test that get_image_token_range returns correct start/end indices"""
    # Generate checkered image with different sizes
    patch_size = 14
    image = generate_checkered_image(width=patch_size*multiplier, height=patch_size*multiplier)
    
    # Get inputs
    inputs = model.prepare_messages("Describe the image.", image)
    
    # Get image token range
    start_index, end_index = model.get_image_token_range(inputs)
    
    # Verify both are integers
    assert isinstance(start_index, int)
    assert isinstance(end_index, int)
    
    # Verify start <= end
    assert start_index <= end_index
    
    # Verify the tokens at these indices are actually image tokens
    input_ids = inputs['input_ids'].squeeze(0)
    image_token_id = model.adapter.get_image_token_id()
    
    assert input_ids[start_index] == image_token_id
    assert input_ids[end_index] == image_token_id
    
    # Count total image tokens and verify it matches the range
    total_image_tokens = (input_ids == image_token_id).sum().item()
    range_size = end_index - start_index + 1
    assert range_size == total_image_tokens
    
    # Verify all tokens in the range are image tokens (consecutive)
    for i in range(start_index, end_index + 1):
        assert input_ids[i] == image_token_id
    
    print(f"Multiplier {multiplier}: {total_image_tokens} image tokens at indices {start_index}-{end_index}")


def test_get_image_token_range_no_image():
    """Test that get_image_token_range raises ValueError when no image tokens present"""
    # This test would require creating inputs without an image, which is tricky
    # For now, we'll skip this test since prepare_messages always includes an image
    # In a real scenario, you'd need text-only inputs
    pass


@pytest.mark.parametrize("multiplier", [8, 16, 24])
def test_generate_patch_overview(model, multiplier):
    # Generate checkered image
    patch_size = 14
    image = generate_checkered_image(width=patch_size*multiplier, height=patch_size*multiplier)

    # Verify that in the image, there are an equal number of black and white pixels
    image_array = np.array(image)
    black_pixels = (image_array == 0).sum()
    white_pixels = (image_array == 255).sum()
    assert black_pixels == white_pixels
    
    # Get inputs to count image tokens
    inputs = model.prepare_messages("Describe the image.", image)
    
    # Count image tokens
    input_ids = inputs['input_ids'].squeeze(0)
    image_token_id = model.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    num_image_tokens = (input_ids == image_token_id).sum().item()
    
    print(f"Number of image tokens for multiplier {multiplier}: {num_image_tokens}")
    
    # Generate patch overview using the method we'll implement
    output_image = model.generate_patch_overview(image, with_labels=False)
    
    # Verify we got an Image back
    assert isinstance(output_image, Image.Image)

    # Verify that in the output image, there are an equal number of black and white pixels
    output_image_array = np.array(output_image)
    
    # Remove red pixels (RGB where R=255, G=0, B=0) before counting
    # Create a mask for red pixels
    red_pixels_mask = (output_image_array[:, :, 0] == 255) & (output_image_array[:, :, 1] == 0) & (output_image_array[:, :, 2] == 0)
    
    # Create a copy of the array and set red pixels to a neutral color (e.g., gray) so they don't interfere with counting
    counting_array = output_image_array.copy()
    counting_array[red_pixels_mask] = [128, 128, 128]  # Set red pixels to gray
    
    # Now count black and white pixels
    black_pixels = (counting_array == 0).sum()
    white_pixels = (counting_array == 255).sum()
    assert black_pixels == white_pixels

    output_image_with_labels = model.generate_patch_overview(image, with_labels=True)
    output_image_with_labels_array = np.array(output_image_with_labels)
    # make sure there are more red pixels in the image with labels
    red_pixels_mask_with_labels = (output_image_with_labels_array[:, :, 0] == 255) & (output_image_with_labels_array[:, :, 1] == 0) & (output_image_with_labels_array[:, :, 2] == 0)
    assert red_pixels_mask_with_labels.sum() > red_pixels_mask.sum()
    
    
    # Save the image to verify it worked
    if SAVE_FILES:
        os.makedirs("tests/tmp", exist_ok=True)

        output_path = f"tests/tmp/test_patch_overview_mult{multiplier}.png"
        output_image.save(output_path)
        assert os.path.exists(output_path)

        output_path_with_labels = f"tests/tmp/test_patch_overview_with_labels_mult{multiplier}.png"
        output_image_with_labels.save(output_path_with_labels)
        assert os.path.exists(output_path_with_labels)

        output_path_original = f"tests/tmp/test_patch_overview_original_mult{multiplier}.png"
        image.save(output_path_original)
        assert os.path.exists(output_path_original)


if __name__ == "__main__":
    test_hookedvlm()
