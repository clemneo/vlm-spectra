"""VLM Spectra - A toolkit for working with Vision-Language Models."""

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hooked_vlm import HookedVLM
from vlm_spectra.core.patch_hooks import (
    HookFn,
    VALID_PATCH_HOOK_TYPES,
    validate_patch_hook_type,
    PatchActivation,
    ZeroAblation,
    AddActivation,
    ScaleActivation,
    PatchHead,
)
from vlm_spectra.models.base_adapter import ModelAdapter
from vlm_spectra.models.registry import ModelRegistry
from vlm_spectra.utils import process_vision_info

__version__ = "0.1.0"
__all__ = [
    "HookedVLM",
    "ActivationCache",
    "ModelRegistry",
    "ModelAdapter",
    # Patch hook utilities
    "HookFn",
    "VALID_PATCH_HOOK_TYPES",
    "validate_patch_hook_type",
    "PatchActivation",
    "ZeroAblation",
    "AddActivation",
    "ScaleActivation",
    "PatchHead",
    "process_vision_info",
]
