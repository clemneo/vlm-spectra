from __future__ import annotations

from abc import ABC, abstractmethod


class Hook(ABC):
    """Base class for all hooks."""

    layer: int

    @abstractmethod
    def __call__(self, module, args, kwargs, output):
        """Apply the hook to a module call."""
