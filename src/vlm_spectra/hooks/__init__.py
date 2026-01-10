"""Hook library for activation capture and patching."""

from .base import Hook
from .capture_hooks import CaptureInputHook, CaptureOutputHook
from .patch_hooks import (
    PatchHeadHook,
    PatchMLPHook,
    PatchResidualHook,
)

__all__ = [
    "Hook",
    "CaptureOutputHook",
    "CaptureInputHook",
    "PatchResidualHook",
    "PatchHeadHook",
    "PatchMLPHook",
]
