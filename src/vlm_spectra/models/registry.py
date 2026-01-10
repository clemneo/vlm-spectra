from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Type

import torch

from vlm_spectra.models.base_adapter import ModelAdapter


class ModelRegistry:
    """Registry for model adapters with lazy auto-discovery."""

    _adapters: Dict[str, Type[ModelAdapter]] = {}
    _model_to_adapter: Dict[str, str] = {}
    _discovered = False

    @classmethod
    def register(cls, adapter_type: str):
        """Decorator to register an adapter class."""

        def decorator(adapter_cls: Type[ModelAdapter]):
            cls._adapters[adapter_type] = adapter_cls
            for model_name in adapter_cls.SUPPORTED_MODELS:
                cls._model_to_adapter[model_name] = adapter_type
            return adapter_cls

        return decorator

    @classmethod
    def _discover_adapters(cls) -> None:
        """Import all adapter modules. Each registers itself if available."""
        if cls._discovered:
            return
        cls._discovered = True

        import vlm_spectra.models.adapters as adapters_pkg

        for _, name, _ in pkgutil.iter_modules(adapters_pkg.__path__):
            try:
                importlib.import_module(f"vlm_spectra.models.adapters.{name}")
            except ImportError:
                pass  # Adapter's dependencies not available, skip silently

    @classmethod
    def get_adapter_class(cls, model_name: str) -> Type[ModelAdapter]:
        cls._discover_adapters()
        adapter_type = cls._model_to_adapter.get(model_name)
        if not adapter_type:
            raise ValueError(f"Model {model_name} not supported")
        return cls._adapters[adapter_type]

    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Return list of all supported model names."""
        cls._discover_adapters()
        return list(cls._model_to_adapter.keys())

    @classmethod
    def load(cls, model_name: str, device: str = "auto", **kwargs):
        """Load model, processor, and create adapter."""
        cls._discover_adapters()
        adapter_cls = cls.get_adapter_class(model_name)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        model = adapter_cls.MODEL_CLASS.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, **kwargs
        )
        processor = adapter_cls.PROCESSOR_CLASS.from_pretrained(model_name)
        adapter = adapter_cls(model)
        adapter.set_processor(processor)
        return model, processor, adapter
