from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hook_manager import HookManager
from vlm_spectra.models.registry import ModelRegistry
from vlm_spectra.preprocessing.prompts import default_prompt_for_model
from vlm_spectra.preprocessing.qwen_processor import QwenProcessor
from vlm_spectra.preprocessing.utils.vision_info import (
    IMAGE_FACTOR,
    MAX_PIXELS,
    MIN_PIXELS,
    process_vision_info,
    smart_resize,
)


class HookedVLM:
    """Main entry point for VLM interpretability."""

    def __init__(
        self,
        model,
        processor: QwenProcessor,
        adapter,
        device: str = "auto",
    ) -> None:
        self.model = model
        self._processor = processor
        self.adapter = adapter
        self._hook_manager = HookManager(adapter)
        self.prompt = processor.default_prompt
        if device == "auto":
            self.device = next(self.model.parameters()).device
        else:
            self.device = device
        self.cache = None

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "auto",
        default_prompt: str | Callable[[str], str] | None = None,
        **kwargs,
    ) -> "HookedVLM":
        if default_prompt is None:
            default_prompt = default_prompt_for_model(model_name)
        model, hf_processor, adapter = ModelRegistry.load(
            model_name, device=device, **kwargs
        )
        processor = QwenProcessor(hf_processor, default_prompt=default_prompt)
        instance = cls(model, processor, adapter, device=device)
        instance.model_name = model_name
        return instance

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 512,
        require_grads: bool = False,
        do_sample: bool = False,
        return_dict_in_generate: bool = True,
        **kwargs,
    ) -> Any:
        inputs = inputs.to(self.device)
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)
        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

        return outputs

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        require_grads: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> Any:
        inputs = inputs.to(self.device)
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)
        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.forward(
                **inputs,
                return_dict=return_dict,
                **kwargs,
            )
        return outputs

    def prepare_inputs(
        self,
        text: str,
        image: Image.Image,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], str]]:
        return self._processor.prepare_inputs(text, image, **kwargs)

    def prepare_messages(
        self,
        task: str,
        image: Image.Image,
        append_text: str = "",
        assistant_prefill: str | None = "",
        return_text: bool = False,
        **kwargs,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], str]]:
        return self.prepare_inputs(
            task,
            image,
            append_text=append_text,
            assistant_prefill=assistant_prefill,
            return_text=return_text,
            **kwargs,
        )

    @contextmanager
    def run_with_cache(self, cache_names: List[str]):
        cache = ActivationCache()
        self._hook_manager.register_cache_hooks(cache, cache_names)
        try:
            yield cache
        finally:
            self._hook_manager.remove_cache_hooks()
            self._hook_manager.finalize_cache(cache, cache_names)
            for key in list(cache.keys()):
                cache[key] = self.adapter.format_cache_item(key[0], cache[key])
            self.cache = cache._data

    @contextmanager
    def run_with_hooks(self, hooks):
        """Context manager for patching activations.

        Hooks should have a `hook_point` attribute specifying where to attach
        and a `layer` attribute specifying which layer to hook.
        """
        self._hook_manager.register_patch_hooks(hooks)
        try:
            yield
        finally:
            self._hook_manager.remove_patch_hooks()

    @property
    def lm_num_layers(self) -> int:
        return self.adapter.lm_num_layers

    @property
    def processor(self):
        return self._processor.processor

    def get_model_components(self):
        return {
            "norm": self.adapter.get_lm_norm(),
            "lm_head": self.adapter.get_lm_head(),
            "tokenizer": self.processor.tokenizer,
        }

    def get_image_token_range(self, inputs) -> tuple[int, int]:
        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = self.adapter.get_image_token_id()

        image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[
            0
        ]

        if len(image_token_positions) == 0:
            raise ValueError("No image tokens found in the input sequence")

        start_index = image_token_positions[0].item()
        end_index = image_token_positions[-1].item()

        return start_index, end_index

    def generate_patch_overview(
        self, image: Image.Image, with_labels: bool = True, start_number: int = 1
    ) -> Image.Image:
        width, height = image.size

        vision_config = self.model.config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * spatial_merge_size,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        merged_grid_h = grid_h // spatial_merge_size
        merged_grid_w = grid_w // spatial_merge_size
        effective_patch_size = patch_size * spatial_merge_size

        resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)

        overlay_image = resized_image.copy()
        draw = ImageDraw.Draw(overlay_image)

        for i in range(merged_grid_h):
            y_start = i * effective_patch_size
            draw.line([(0, y_start), (resized_width, y_start)], fill="red", width=1)
            y_end = (i + 1) * effective_patch_size - 1
            draw.line([(0, y_end), (resized_width, y_end)], fill="red", width=1)

        for j in range(merged_grid_w):
            x_start = j * effective_patch_size
            draw.line([(x_start, 0), (x_start, resized_height)], fill="red", width=1)
            x_end = (j + 1) * effective_patch_size - 1
            draw.line([(x_end, 0), (x_end, resized_height)], fill="red", width=1)

        font_size = max(14, min(32, effective_patch_size // 3))

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            try:
                font = ImageFont.load_default()
            except (OSError, IOError):
                font = None

        if with_labels:
            patch_num = start_number
            for i in range(merged_grid_h):
                for j in range(merged_grid_w):
                    if patch_num % 10 == 1 or patch_num == merged_grid_h * merged_grid_w:
                        x = j * effective_patch_size + effective_patch_size // 4
                        y = i * effective_patch_size + effective_patch_size // 4

                        text = str(patch_num)
                        if font:
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        else:
                            text_width, text_height = len(text) * 6, 10

                        draw.rectangle(
                            [x - 1, y - 1, x + text_width + 1, y + text_height + 4],
                            fill="white",
                            outline="red",
                        )

                        draw.text((x, y), text, fill="red", font=font)

                    patch_num += 1

        return overlay_image

    def prepare_messages_batch(
        self,
        tasks: List[str],
        images: List[Image.Image],
        append_text: str = "",
        assistant_prefill: str | None = "",
        return_text: bool = False,
    ):
        if len(tasks) != len(images):
            raise ValueError(
                f"Number of tasks ({len(tasks)}) must match number of images ({len(images)})"
            )

        assistant_prefill = "" if assistant_prefill is None else assistant_prefill

        batch_messages = []
        batch_texts = []

        for task, image in zip(tasks, images):
            prompt_text = self._processor._render_prompt(task, None)
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
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, continue_final_message=True
                )
            else:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            if append_text:
                text += append_text

            batch_messages.append(messages)
            batch_texts.append(text)

        all_image_inputs = []
        all_video_inputs = []

        for messages in batch_messages:
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
            if video_inputs:
                all_video_inputs.extend(video_inputs)

        inputs = self.processor(
            text=batch_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        if return_text:
            return inputs, batch_texts
        return inputs

    def forward_batch(
        self,
        inputs_list=None,
        tasks=None,
        images=None,
        require_grads: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        if inputs_list is None:
            if tasks is None or images is None:
                raise ValueError(
                    "Either inputs_list or both tasks and images must be provided"
                )
            inputs = self.prepare_messages_batch(tasks, images)
        else:
            if isinstance(inputs_list, list) and len(inputs_list) == 1:
                inputs = inputs_list[0]
            else:
                raise NotImplementedError(
                    "Complex collation of input_list not yet implemented. Use tasks/images instead."
                )

        inputs = inputs.to(self.device)
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)

        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.forward(
                **inputs,
                return_dict=return_dict,
                **kwargs,
            )
        return outputs

    def generate_batch(
        self,
        inputs_list=None,
        tasks=None,
        images=None,
        max_new_tokens: int = 512,
        require_grads: bool = False,
        do_sample: bool = False,
        return_dict_in_generate: bool = True,
        **kwargs,
    ):
        if inputs_list is None:
            if tasks is None or images is None:
                raise ValueError(
                    "Either inputs_list or both tasks and images must be provided"
                )
            inputs = self.prepare_messages_batch(tasks, images)
        else:
            if isinstance(inputs_list, list) and len(inputs_list) == 1:
                inputs = inputs_list[0]
            else:
                raise NotImplementedError(
                    "Complex collation of input_list not yet implemented. Use tasks/images instead."
                )

        inputs = inputs.to(self.device)
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)

        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                return_dict_in_generate=return_dict_in_generate,
                **kwargs,
            )

        return outputs
