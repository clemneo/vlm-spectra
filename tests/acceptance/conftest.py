"""Fixtures for acceptance tests.

Acceptance tests run against real models to verify core contracts.
These tests are slower and require GPU for larger models.
"""

import gc

import pytest
import torch

from vlm_spectra import HookedVLM
from vlm_spectra.models.registry import ModelRegistry


# Model capabilities for capability-gated tests
MODEL_CAPABILITIES = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": {
        "contiguous_image_tokens": False,
        "supports_batching": True,
        "supports_flash_attn_patterns": True,
    },
    "HuggingFaceTB/SmolVLM-500M-Instruct": {
        "contiguous_image_tokens": False,
        "supports_batching": True,
        "supports_flash_attn_patterns": True,
    },
    "HuggingFaceTB/SmolVLM-Instruct": {
        "contiguous_image_tokens": False,
        "supports_batching": True,
        "supports_flash_attn_patterns": True,
    },
    "ByteDance-Seed/UI-TARS-1.5-7B": {
        "contiguous_image_tokens": True,
        "supports_batching": True,
        "supports_flash_attn_patterns": True,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "contiguous_image_tokens": True,
        "supports_batching": True,
        "supports_flash_attn_patterns": True,
    },
}

# Model aliases for convenience
MODEL_ALIASES = {
    "uitars1.5-7b": "ByteDance-Seed/UI-TARS-1.5-7B",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-2b": "HuggingFaceTB/SmolVLM-Instruct",
}


def pytest_addoption(parser):
    """Add --model option for selecting specific model."""
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model name or alias to run tests against",
    )
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run manual verification tests",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_capability(name): mark test as requiring specific model capability"
    )


def pytest_collection_modifyitems(config, items):
    """Handle manual test marker."""
    if not config.getoption("--run-manual"):
        skip_manual = pytest.mark.skip(reason="need --run-manual option to run")
        for item in items:
            if "manual" in item.keywords:
                item.add_marker(skip_manual)


def pytest_runtest_setup(item):
    """Skip tests if model lacks required capability."""
    for marker in item.iter_markers(name="requires_capability"):
        cap_name = marker.args[0]

        # Get the model from the fixture
        model = None
        if "model" in item.fixturenames:
            # Try to get model from fixture values
            try:
                model = item.funcargs.get("model")
            except (AttributeError, KeyError):
                pass

        if model is not None:
            model_name = getattr(model, "model_name", None)
            capabilities = MODEL_CAPABILITIES.get(model_name, {})
            if not capabilities.get(cap_name, False):
                pytest.skip(f"Model {model_name} lacks capability: {cap_name}")


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically clean up GPU memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Get available models from registry
MODEL_NAMES = ModelRegistry.list_supported_models()


@pytest.fixture(scope="session", params=MODEL_NAMES)
def model(request):
    """Load a HookedVLM model.

    Session-scoped so all tests for one model run before moving to the next,
    preventing OOM from multiple models.
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
def capabilities(model):
    """Get capabilities for current model."""
    model_name = getattr(model, "model_name", None)
    return MODEL_CAPABILITIES.get(model_name, {})
