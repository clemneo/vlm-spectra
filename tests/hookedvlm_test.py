import torch
from vlm_spectra.models.HookedVLM import HookedVLM
import pytest

from PIL import Image
import numpy as np

MODEL_NAMES = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]


@pytest.fixture(scope="session", params=MODEL_NAMES)
def model(request):
    return HookedVLM(request.param)


def generate_random_image(width=224, height=224, num_channels=3):
    random_array = np.random.randint(
        0, 255, (height, width, num_channels), dtype=np.uint8
    )
    random_image = Image.fromarray(random_array)
    return random_image


def test_hookedvlm(model):
    assert model is not None


def test_generate(model):
    image = generate_random_image()
    outputs = model.generate("Describe the image.", image)

    assert type(outputs.sequences) is torch.Tensor


def test_generate_with_output_hidden_states(model):
    image = generate_random_image()
    outputs = model.generate("Describe the image.", image, output_hidden_states=True)

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
    outputs = model.forward("Describe the image.", image)
    assert type(outputs.logits) is torch.Tensor


def test_forward_with_output_hidden_states(model):
    image = generate_random_image()
    outputs = model.forward("Describe the image.", image, output_hidden_states=True)

    assert type(outputs.logits) is torch.Tensor

    assert type(outputs.hidden_states) is tuple  # tuple of layers
    assert len(outputs.hidden_states) > 0

    assert (
        type(outputs.hidden_states[0]) is torch.Tensor
    )  # the tensor of shape (batch_size, sequence_length, hidden_size)


def test_get_model_components(model):
    components = model.get_model_components()
    assert type(components) is dict
    assert len(components) > 0


if __name__ == "__main__":
    test_hookedvlm()
