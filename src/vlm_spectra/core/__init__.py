"""Core VLM components."""

from .activation_cache import ActivationCache
from .hook_manager import HookManager
from .hooked_vlm import HookedVLM
from .patch_hooks import (
    HookFn,
    VALID_PATCH_HOOK_TYPES,
    validate_patch_hook_type,
    PatchActivation,
    ZeroAblation,
    AddActivation,
    ScaleActivation,
    PatchHead,
)

__all__ = [
    "ActivationCache",
    "HookManager",
    "HookedVLM",
    # Patch hook utilities
    "HookFn",
    "VALID_PATCH_HOOK_TYPES",
    "validate_patch_hook_type",
    "PatchActivation",
    "ZeroAblation",
    "AddActivation",
    "ScaleActivation",
    "PatchHead",
]
