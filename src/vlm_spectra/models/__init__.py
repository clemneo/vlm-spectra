"""Models subpackage for VLM Spectra."""

__all__ = ["HookedVLM"]


def __getattr__(name: str):
    if name == "HookedVLM":
        from vlm_spectra.models.HookedVLM import HookedVLM

        return HookedVLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
