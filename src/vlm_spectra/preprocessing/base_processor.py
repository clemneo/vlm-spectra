from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

if TYPE_CHECKING:
    from vlm_spectra.preprocessing.spatial import ImageInfo


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

    @abstractmethod
    def prepare_inputs_batch(
        self,
        tasks: List[str],
        images: List[Image.Image],
        prompt_template: Optional[str] = None,
        append_text: str = "",
        assistant_prefill: Optional[str] = "",
        return_text: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[str]]]:
        """Convert a batch of tasks + images to model inputs."""

    @abstractmethod
    def process_image(self, image: Image.Image) -> ImageInfo:
        """Process an image and return spatial metadata."""
