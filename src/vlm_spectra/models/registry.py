from __future__ import annotations

from typing import Dict, Type

import torch

from vlm_spectra.models.base_adapter import ModelAdapter


class ModelRegistry:
    """Registry for model adapters."""

    _adapters: Dict[str, Type[ModelAdapter]] = {}
    _model_to_adapter: Dict[str, str] = {}

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
    def get_adapter_class(cls, model_name: str) -> Type[ModelAdapter]:
        adapter_type = cls._model_to_adapter.get(model_name)
        if not adapter_type:
            raise ValueError(f"Model {model_name} not supported")
        return cls._adapters[adapter_type]

    @classmethod
    def load(cls, model_name: str, device: str = "auto", **kwargs):
        """Load model, processor, and create adapter."""
        adapter_cls = cls.get_adapter_class(model_name)
        torch_dtype = kwargs.pop("torch_dtype", torch.bfloat16)
        model = adapter_cls.MODEL_CLASS.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device, **kwargs
        )
        processor = adapter_cls.PROCESSOR_CLASS.from_pretrained(model_name)
        adapter = adapter_cls(model)
        adapter.set_processor(processor)
        return model, processor, adapter


from vlm_spectra.models.adapters import qwen25_vl  # noqa: F401,E402
