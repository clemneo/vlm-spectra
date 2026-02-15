from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.preprocessing.base_processor import BaseProcessor
from vlm_spectra.preprocessing.spatial import ImageInfo
from vlm_spectra.preprocessing.utils.vision_info import (
    MAX_PIXELS,
    MIN_PIXELS,
    process_vision_info,
    smart_resize,
)


class QwenProcessor(BaseProcessor):
    """Qwen2.5-VL preprocessing wrapper."""

    def __init__(
        self,
        hf_processor,
        default_prompt: Optional[Union[str, Callable[[str], str]]] = None,
        image_factor: Optional[int] = None,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
    ) -> None:
        self.processor = hf_processor
        self.default_prompt = default_prompt
        self.image_factor = image_factor
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

    def prepare_inputs(
        self,
        text: str,
        image: Image.Image,
        prompt_template: Optional[str] = None,
        append_text: str = "",
        assistant_prefill: Optional[str] = "",
        return_text: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], str]]:
        assistant_prefill = "" if assistant_prefill is None else assistant_prefill
        prompt_text = self._render_prompt(text, prompt_template)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": image},
                ],
            }
        ]

        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        if assistant_prefill:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )
        else:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if append_text:
            rendered_text += append_text

        image_inputs, video_inputs = process_vision_info(
            messages, image_factor=self.image_factor
        )
        inputs = self.processor(
            text=[rendered_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if return_text:
            return inputs, rendered_text
        return inputs

    def process_image(self, image: Image.Image) -> ImageInfo:
        """Process an image and return spatial metadata."""
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        factor = self.patch_size * self.spatial_merge_size
        resized_h, resized_w = smart_resize(
            orig_h, orig_w, factor=factor, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

        processed = image.resize((resized_w, resized_h), Image.LANCZOS)
        grid_h = resized_h // factor
        grid_w = resized_w // factor

        return ImageInfo(
            image=processed,
            original_size=(orig_w, orig_h),
            processed_size=(resized_w, resized_h),
            crop_box=None,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            grid_h=grid_h,
            grid_w=grid_w,
        )

    def _render_prompt(
        self,
        text: str,
        prompt_template: Optional[Union[str, Callable[[str], str]]],
    ) -> str:
        template = prompt_template or self.default_prompt
        if not template:
            return text
        if callable(template):
            return template(text)
        try:
            return template.format(language="English", instruction=text, text=text)
        except KeyError:
            return text
