import torch
from vlm_spectrum.models.HookedVLM import HookedVLM
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
    random_array = np.random.randint(0, 255, (height, width, num_channels), dtype=np.uint8)
    random_image = Image.fromarray(random_array)
    return random_image


def test_hookedvlm(model):
    assert model is not None

def test_generate(model):
    image = generate_random_image()
    outputs = model.generate("Describe the image.", image)

    assert type(outputs.sequences) is torch.Tensor
    assert outputs.sequences.shape[0] == 1
    assert outputs.sequences.shape[1] > 0


if __name__ == "__main__":
    test_hookedvlm()