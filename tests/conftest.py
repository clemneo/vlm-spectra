"""Shared test fixtures and utilities for VLM Spectra test suite.

This file contains only utilities shared across test tiers.
Model fixtures are defined in their respective test directories:
- integration/conftest.py - tiny_model fixture
- acceptance/conftest.py - full model fixture
"""

import numpy as np
from PIL import Image


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
