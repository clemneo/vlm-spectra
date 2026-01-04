"""Backward compatibility - use vlm_spectra.HookedVLM instead."""

import warnings

from vlm_spectra.core.hooked_vlm import HookedVLM as NewHookedVLM
from vlm_spectra.preprocessing.qwen_processor import UI_TARS_PROMPT


class HookedVLM(NewHookedVLM):
    def __init__(
        self, model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B", device: str = "auto"
    ) -> None:
        warnings.warn(
            "vlm_spectra.models.HookedVLM is deprecated. "
            "Use vlm_spectra.HookedVLM.from_pretrained() instead.",
            DeprecationWarning,
        )
        instance = NewHookedVLM.from_pretrained(
            model_name, device=device, default_prompt=UI_TARS_PROMPT
        )
        self.__dict__.update(instance.__dict__)
