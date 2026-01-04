"""Hook library for activation capture and patching."""

from .base import Hook
from .capture_hooks import CaptureInputHook, CaptureOutputHook
from .patch_hooks import (
    MeanAblationHook,
    PatchHeadHook,
    PatchMLPHook,
    PatchResidualHook,
    ZeroAblationHook,
)

__all__ = [
    "Hook",
    "CaptureOutputHook",
    "CaptureInputHook",
    "PatchResidualHook",
    "PatchHeadHook",
    "PatchMLPHook",
    "ZeroAblationHook",
    "MeanAblationHook",
]
