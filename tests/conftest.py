"""Shared test fixtures and utilities for VLM Spectra test suite."""

import gc

import pytest
import numpy as np
import torch
from PIL import Image

from vlm_spectra import HookedVLM
from vlm_spectra.models.registry import ModelRegistry


# Get available models from registry (triggers lazy discovery)
# Only models whose dependencies are available will be listed
MODEL_NAMES = ModelRegistry.list_supported_models()

MODEL_ALIASES = {
    "uitars1.5-7b": "ByteDance-Seed/UI-TARS-1.5-7B",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
}


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model name or alias to run tests against",
    )


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically clean up GPU memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session", params=MODEL_NAMES)
def model(request):
    """Load a HookedVLM model. Session-scoped so all tests for one model
    run before moving to the next, preventing OOM from multiple models.

    Explicitly cleans up GPU memory after each model to prevent OOM
    when testing multiple models sequentially.
    """
    selected = request.config.getoption("--model")
    if selected:
        selected = MODEL_ALIASES.get(selected, selected)
        if request.param != selected:
            pytest.skip(f"Skipping {request.param}; --model={selected}")

    loaded_model = HookedVLM.from_pretrained(request.param)
    yield loaded_model

    # Cleanup: explicitly delete model and free GPU memory
    del loaded_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def random_image():
    """Generate a random RGB image for testing."""
    return generate_random_image()


@pytest.fixture
def checkered_image():
    """Generate a checkered pattern image for deterministic testing."""
    return generate_checkered_image()


@pytest.fixture
def small_image():
    """Generate a small image for memory-efficient gradient tests."""
    return generate_random_image(width=28, height=28)


def generate_random_image(width=56, height=56, num_channels=3, seed=None):
    """Generate a random RGB image.

    Args:
        width: Image width in pixels (default 56 for memory efficiency)
        height: Image height in pixels (default 56 for memory efficiency)
        num_channels: Number of color channels (3 for RGB)
        seed: Optional random seed for reproducibility

    Returns:
        PIL.Image: Random RGB image
    """
    if seed is not None:
        np.random.seed(seed)
    random_array = np.random.randint(
        0, 255, (height, width, num_channels), dtype=np.uint8
    )
    return Image.fromarray(random_array)


def generate_checkered_image(width=56, height=56, checkered_size=14, seed=None):
    """Generate a black and white checkered pattern image.

    Useful for deterministic testing where pixel patterns matter.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        checkered_size: Size of each square in the checkerboard
        seed: Optional random seed (unused, for API compatibility)

    Returns:
        PIL.Image: Checkered pattern RGB image
    """
    _ = seed  # unused, kept for API compatibility
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


def get_patch_size(model):
    """Get the vision patch size from model config."""
    return model.model.config.vision_config.patch_size
