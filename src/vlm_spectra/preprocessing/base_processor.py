from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
from PIL import Image


class BaseProcessor(ABC):
    """Abstract base for model-specific preprocessing."""

    @abstractmethod
    def prepare_inputs(
        self,
        text: str,
        image: Image.Image,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convert text + image to model inputs."""
