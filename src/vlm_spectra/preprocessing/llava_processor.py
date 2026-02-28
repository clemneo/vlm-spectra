from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.preprocessing.base_processor import BaseProcessor

if TYPE_CHECKING:
    from vlm_spectra.preprocessing.spatial import ImageInfo


class LlavaProcessor(BaseProcessor):
    """LLaVA preprocessing wrapper."""

    def __init__(
        self,
        hf_processor,
        default_prompt: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> None:
        self.processor = hf_processor
        self.default_prompt = default_prompt
        self.image_factor = None
        self.patch_size = 14
        self.spatial_merge_size = 1

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
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        rendered_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if assistant_prefill:
            rendered_text += assistant_prefill

        if append_text:
            rendered_text += append_text

        inputs = self.processor(
            text=rendered_text,
            images=[image],
            return_tensors="pt",
        )

        if return_text:
            return inputs, rendered_text
        return inputs

    def prepare_inputs_batch(
        self,
        tasks: List[str],
        images: List[Image.Image],
        prompt_template: Optional[str] = None,
        append_text: str = "",
        assistant_prefill: Optional[str] = "",
        return_text: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[str]]]:
        if len(tasks) != len(images):
            raise ValueError(
                f"Number of tasks ({len(tasks)}) must match number of images ({len(images)})"
            )
        assistant_prefill = "" if assistant_prefill is None else assistant_prefill

        batch_texts = []
        for task in tasks:
            prompt_text = self._render_prompt(task, prompt_template)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if assistant_prefill:
                rendered_text += assistant_prefill

            if append_text:
                rendered_text += append_text

            batch_texts.append(rendered_text)

        inputs = self.processor(
            text=batch_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        if return_text:
            return inputs, batch_texts
        return inputs

    def process_image(self, image: Image.Image) -> "ImageInfo":
        from vlm_spectra.preprocessing.spatial import ImageInfo

        image = image.convert("RGB")
        orig_w, orig_h = image.size

        target_size = 224
        patch_size = 14
        spatial_merge_size = 1

        # Step 1: Resize shortest edge to 224, preserving aspect ratio
        if orig_w <= orig_h:
            new_w = target_size
            new_h = int(round(orig_h * target_size / orig_w))
        else:
            new_h = target_size
            new_w = int(round(orig_w * target_size / orig_h))

        resized = image.resize((new_w, new_h), Image.LANCZOS)

        # Step 2: Center crop to 224x224
        crop_left = (new_w - target_size) // 2
        crop_top = (new_h - target_size) // 2
        processed = resized.crop(
            (crop_left, crop_top, crop_left + target_size, crop_top + target_size)
        )

        # Step 3: Compute crop_box in original image coordinates
        # Map the crop offsets back to original space
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        crop_box_x1 = int(round(crop_left * scale_x))
        crop_box_y1 = int(round(crop_top * scale_y))
        crop_box_x2 = int(round((crop_left + target_size) * scale_x))
        crop_box_y2 = int(round((crop_top + target_size) * scale_y))

        # If no actual cropping occurred, set crop_box to None
        crop_box: Optional[Tuple[int, int, int, int]] = None
        if crop_left > 0 or crop_top > 0:
            crop_box = (crop_box_x1, crop_box_y1, crop_box_x2, crop_box_y2)

        grid_size = target_size // (patch_size * spatial_merge_size)

        return ImageInfo(
            image=processed,
            original_size=(orig_w, orig_h),
            processed_size=(target_size, target_size),
            crop_box=crop_box,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            grid_h=grid_size,
            grid_w=grid_size,
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
            return template.format(text=text, instruction=text)
        except KeyError:
            return text
