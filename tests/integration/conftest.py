"""Fixtures for integration tests.

Integration tests load a small model (SmolVLM-256M) to test processor behavior
and hook registration without the overhead of larger models.
"""

import pytest

from vlm_spectra import HookedVLM


@pytest.fixture(scope="module")
def tiny_model():
    """SmolVLM-256M for integration tests - smallest available.

    Module-scoped to avoid reloading for each test.
    """
    return HookedVLM.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")


@pytest.fixture(scope="module")
def processor_only():
    """Load processor without full model weights.

    Useful for testing tokenization/preprocessing in isolation.
    """
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
