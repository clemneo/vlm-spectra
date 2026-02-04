from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.preprocessing.base_processor import BaseProcessor


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
