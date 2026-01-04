"""VLM Spectra - A toolkit for working with Vision-Language Models."""

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hooked_vlm import HookedVLM
from vlm_spectra.models.base_adapter import ModelAdapter
from vlm_spectra.models.registry import ModelRegistry
from vlm_spectra.hooks import Hook, PatchHeadHook, PatchResidualHook, ZeroAblationHook
from vlm_spectra.utils import process_vision_info

__version__ = "0.1.0"
__all__ = [
    "HookedVLM",
    "ActivationCache",
    "ModelRegistry",
    "ModelAdapter",
    "Hook",
    "PatchResidualHook",
    "PatchHeadHook",
    "ZeroAblationHook",
    "process_vision_info",
]
