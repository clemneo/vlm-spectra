"""Model-specific adapters."""

from .qwen25_vl import Qwen25VLAdapter
from .qwen3_vl import Qwen3VLAdapter

__all__ = ["Qwen25VLAdapter", "Qwen3VLAdapter"]
