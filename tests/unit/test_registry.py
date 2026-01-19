"""Unit tests for ModelRegistry class.

These tests validate the ModelRegistry lookup and registration functionality
without loading any actual models.
All tests should run in < 1 second total.
"""

import pytest

from vlm_spectra.models.registry import ModelRegistry


class TestModelRegistryLookup:
    """Test ModelRegistry lookup functionality."""

    def test_list_supported_models_returns_list(self):
        """list_supported_models should return a list."""
        models = ModelRegistry.list_supported_models()
        assert isinstance(models, list)

    def test_list_supported_models_non_empty(self):
        """list_supported_models should return non-empty list."""
        models = ModelRegistry.list_supported_models()
        assert len(models) > 0, "No models registered"

    def test_list_supported_models_contains_strings(self):
        """list_supported_models should return list of strings."""
        models = ModelRegistry.list_supported_models()
        assert all(isinstance(m, str) for m in models)

    def test_get_adapter_class_returns_class(self):
        """get_adapter_class should return a class for known model."""
        models = ModelRegistry.list_supported_models()
        if not models:
            pytest.skip("No models registered")

        adapter_cls = ModelRegistry.get_adapter_class(models[0])
        assert adapter_cls is not None
        assert isinstance(adapter_cls, type)

    def test_unknown_model_raises(self):
        """get_adapter_class should raise ValueError for unknown model."""
        with pytest.raises(ValueError, match="not supported"):
            ModelRegistry.get_adapter_class("definitely-not-a-real-model/fake-123")

    def test_known_models_have_adapters(self):
        """All listed models should have adapter classes."""
        models = ModelRegistry.list_supported_models()
        for model_name in models:
            adapter_cls = ModelRegistry.get_adapter_class(model_name)
            assert adapter_cls is not None, f"No adapter for {model_name}"


class TestModelRegistryAdapters:
    """Test adapter class properties."""

    def test_adapter_has_supported_models(self):
        """Adapter classes should have SUPPORTED_MODELS attribute."""
        models = ModelRegistry.list_supported_models()
        if not models:
            pytest.skip("No models registered")

        adapter_cls = ModelRegistry.get_adapter_class(models[0])
        assert hasattr(adapter_cls, "SUPPORTED_MODELS")
        assert isinstance(adapter_cls.SUPPORTED_MODELS, (list, tuple))

    def test_adapter_has_model_class(self):
        """Adapter classes should have MODEL_CLASS attribute."""
        models = ModelRegistry.list_supported_models()
        if not models:
            pytest.skip("No models registered")

        adapter_cls = ModelRegistry.get_adapter_class(models[0])
        assert hasattr(adapter_cls, "MODEL_CLASS")

    def test_adapter_has_processor_class(self):
        """Adapter classes should have PROCESSOR_CLASS attribute."""
        models = ModelRegistry.list_supported_models()
        if not models:
            pytest.skip("No models registered")

        adapter_cls = ModelRegistry.get_adapter_class(models[0])
        assert hasattr(adapter_cls, "PROCESSOR_CLASS")


class TestModelRegistryDiscovery:
    """Test adapter auto-discovery."""

    def test_discovery_is_idempotent(self):
        """Multiple discovery calls should return same results."""
        models1 = ModelRegistry.list_supported_models()
        models2 = ModelRegistry.list_supported_models()
        assert set(models1) == set(models2)

    def test_smolvlm_in_supported_models(self):
        """SmolVLM should be in supported models if dependencies are available."""
        models = ModelRegistry.list_supported_models()
        # SmolVLM models should be available
        smolvlm_models = [m for m in models if "SmolVLM" in m]
        assert len(smolvlm_models) > 0, "No SmolVLM models found"
