"""Fixtures for acceptance tests."""

import gc
import pytest
import torch

from vlm_spectra import HookedVLM
from vlm_spectra.models.registry import ModelRegistry


# Default model when neither --model nor --all-models is specified
DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

MODEL_CAPABILITIES = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": {
        "contiguous_image_tokens": False,
        "supports_batching": True,
    },
    "HuggingFaceTB/SmolVLM-500M-Instruct": {
        "contiguous_image_tokens": False,
        "supports_batching": True,
    },
    "HuggingFaceTB/SmolVLM-Instruct": {
        "contiguous_image_tokens": False,
        "supports_batching": True,
    },
    "ByteDance-Seed/UI-TARS-1.5-7B": {
        "contiguous_image_tokens": True,
        "supports_batching": True,
    },
    "Qwen/Qwen3-VL-8B-Instruct": {
        "contiguous_image_tokens": True,
        "supports_batching": True,
    },
}

MODEL_ALIASES = {
    "uitars1.5-7b": "ByteDance-Seed/UI-TARS-1.5-7B",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "smolvlm-256m": "HuggingFaceTB/SmolVLM-256M-Instruct",
    "smolvlm-500m": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "smolvlm-2b": "HuggingFaceTB/SmolVLM-Instruct",
}


def pytest_addoption(parser):
    """Add --model, --all-models, and --run-manual options."""
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Model name or alias to run tests against (default: smolvlm-256m)",
    )
    parser.addoption(
        "--all-models",
        action="store_true",
        default=False,
        help="Run tests against all registered models",
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
    """Handle manual tests."""
    if not config.getoption("--run-manual"):
        skip_manual = pytest.mark.skip(reason="need --run-manual option to run")
        for item in items:
            if "manual" in item.keywords:
                item.add_marker(skip_manual)


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically clean up GPU memory after each test."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_models_to_test(config):
    """Determine which models to test based on CLI options."""
    selected = config.getoption("--model")
    all_models = config.getoption("--all-models")

    if all_models:
        # Run against all registered models
        return ModelRegistry.list_supported_models()
    elif selected:
        # Run against specific model
        resolved = MODEL_ALIASES.get(selected, selected)
        return [resolved]
    else:
        # Default: only SmolVLM-256M
        return [DEFAULT_MODEL]


def pytest_generate_tests(metafunc):
    """Dynamically parameterize the model fixture based on CLI options."""
    if "model" in metafunc.fixturenames:
        models = get_models_to_test(metafunc.config)
        metafunc.parametrize("model", models, indirect=True, scope="session")


@pytest.fixture(scope="session")
def model(request):
    """Load a HookedVLM model."""
    model_name = request.param
    loaded_model = HookedVLM.from_pretrained(model_name)
    yield loaded_model

    del loaded_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def check_capability(request, model):
    """Skip tests if model lacks required capability.

    This runs AFTER model is loaded, so we can properly access it.
    """
    for marker in request.node.iter_markers(name="requires_capability"):
        cap_name = marker.args[0]
        model_name = getattr(model, "model_name", None)
        capabilities = MODEL_CAPABILITIES.get(model_name, {})
        if not capabilities.get(cap_name, False):
            pytest.skip(f"Model {model_name} lacks capability: {cap_name}")


@pytest.fixture
def capabilities(model):
    """Get capabilities for current model."""
    model_name = getattr(model, "model_name", None)
    return MODEL_CAPABILITIES.get(model_name, {})
