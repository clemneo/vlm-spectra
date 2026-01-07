"""Backward-compatible adapter entry point."""

from transformers import (
    Idefics3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from vlm_spectra.models.adapters.qwen25_vl import Qwen25VLAdapter
from vlm_spectra.models.adapters.qwen3_vl import Qwen3VLAdapter
from vlm_spectra.models.adapters.smolvlm import SmolVLMAdapter
from vlm_spectra.models.base_adapter import ModelAdapter


def get_model_adapter(model) -> ModelAdapter:
    if isinstance(model, Qwen2_5_VLForConditionalGeneration):
        return Qwen25VLAdapter(model)
    if isinstance(model, Qwen3VLForConditionalGeneration):
        return Qwen3VLAdapter(model)
    if isinstance(model, Idefics3ForConditionalGeneration):
        return SmolVLMAdapter(model)
    raise ValueError(f"Model {model} not supported")


__all__ = ["ModelAdapter", "Qwen25VLAdapter", "Qwen3VLAdapter", "SmolVLMAdapter", "get_model_adapter"]
