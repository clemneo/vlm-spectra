from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.preprocessing.base_processor import BaseProcessor
from vlm_spectra.preprocessing.utils.vision_info import process_vision_info


class QwenProcessor(BaseProcessor):
    """Qwen2.5-VL preprocessing wrapper."""

    def __init__(
        self,
        hf_processor,
        default_prompt: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> None:
        self.processor = hf_processor
        self.default_prompt = default_prompt

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
            try:
                rendered_text = self.processor.apply_chat_template(
                    messages, tokenize=False, continue_final_message=True
                )
            except ValueError:
                # Some templates don't support continue_final_message; fall back to generation prompt + manual prefill.
                rendered_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                rendered_text += assistant_prefill
        else:
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if append_text:
            rendered_text += append_text

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[rendered_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Provide a lightweight image_grid_thw if the processor doesn't return one (e.g., SmolVLM).
        if "image_grid_thw" not in inputs and "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            # pixel_values shape: (batch, num_images, 3, H, W)
            _, num_images, _, height, width = pv.shape
            patch = getattr(getattr(self.processor, "vision_config", None), "patch_size", None)
            if patch is None and hasattr(self.processor, "image_processor"):
                patch = getattr(self.processor.image_processor, "patch_size", None)
            patch = patch or 1
            grid_h = height // patch
            grid_w = width // patch
            inputs["image_grid_thw"] = torch.tensor(
                [[num_images, grid_h, grid_w]], dtype=torch.int64
            )

        if return_text:
            return inputs, rendered_text
        return inputs

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
