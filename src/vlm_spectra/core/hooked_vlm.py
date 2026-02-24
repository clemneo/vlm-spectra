from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.core.activation_cache import ActivationCache
from vlm_spectra.core.hook_manager import HookManager
from vlm_spectra.core.hook_points import HookPoint
from vlm_spectra.models.registry import ModelRegistry
from vlm_spectra.preprocessing.prompts import default_prompt_for_model
from vlm_spectra.preprocessing.llava_processor import LlavaProcessor
from vlm_spectra.preprocessing.base_processor import BaseProcessor
from vlm_spectra.preprocessing.qwen_processor import QwenProcessor
from vlm_spectra.preprocessing.smolvlm_processor import SmolVLMProcessor
from vlm_spectra.preprocessing.spatial import ImageInfo
from vlm_spectra.preprocessing.utils.vision_info import (
    IMAGE_FACTOR,
    resolve_patch_params,
)


class HookedVLM:
    """Main entry point for VLM interpretability."""

    def __init__(
        self,
        model,
        processor: BaseProcessor,
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
        self.image_factor = self._resolve_image_factor(model)
        if getattr(self._processor, "image_factor", None) is None:
            self._processor.image_factor = self.image_factor
        self.cache = None

    @staticmethod
    def _resolve_image_factor(model) -> int:
        config = getattr(model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        model_name = getattr(config, "name_or_path", None)
        if vision_config is None:
            return IMAGE_FACTOR
        patch_size, spatial_merge_size = resolve_patch_params(vision_config, model_name)
        return patch_size * spatial_merge_size

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
        if "SmolVLM" in model_name:
            processor = SmolVLMProcessor(hf_processor, default_prompt=default_prompt)
        elif "llava" in model_name.lower():
            processor = LlavaProcessor(hf_processor, default_prompt=default_prompt)
        else:
            config = getattr(model, "config", None)
            vision_config = getattr(config, "vision_config", None)
            model_name_cfg = getattr(config, "name_or_path", None)
            patch_size, spatial_merge_size = resolve_patch_params(
                vision_config, model_name_cfg
            )
            image_factor = patch_size * spatial_merge_size
            processor = QwenProcessor(
                hf_processor,
                default_prompt=default_prompt,
                image_factor=image_factor,
                patch_size=patch_size,
                spatial_merge_size=spatial_merge_size,
            )
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
        needs_eager = kwargs.get("output_attentions", False)
        if needs_eager and hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        try:
            with torch.no_grad() if not require_grads else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    return_dict_in_generate=return_dict_in_generate,
                    **kwargs,
                )
        finally:
            if needs_eager and hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation(self.adapter._original_attn_impl)
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
        needs_eager = kwargs.get("output_attentions", False)
        if needs_eager and hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        try:
            with torch.no_grad() if not require_grads else nullcontext():
                outputs = self.model.forward(
                    **inputs,
                    return_dict=return_dict,
                    **kwargs,
                )
        finally:
            if needs_eager and hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation(self.adapter._original_attn_impl)
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

    def process_image(self, image: Image.Image) -> ImageInfo:
        """Process an image and return spatial metadata with correction utilities."""
        return self._processor.process_image(image)

    @contextmanager
    def run_with_cache(self, cache_names: List[str]):
        """Context manager for capturing activations.

        Args:
            cache_names: List of hook names to capture. Supports wildcards.
                Examples:
                    ["lm.blocks.*.hook_resid_post"]  # all layers
                    ["lm.blocks.5.hook_resid_post"]  # specific layer
                    ["lm.blocks.*.attn.hook_pattern"]  # attention patterns

        Yields:
            ActivationCache with captured activations accessible by string keys.

        Example:
            with model.run_with_cache(["lm.blocks.*.hook_resid_post"]) as cache:
                model.forward(inputs)
            residual = cache["lm.blocks.5.hook_resid_post"]
            stacked = cache.stack("lm.blocks.*.hook_resid_post")
        """
        cache = ActivationCache()
        self._hook_manager.register_cache_hooks(cache, cache_names)
        try:
            yield cache
        finally:
            self._hook_manager.finalize_cache(cache, cache_names)
            self._hook_manager.remove_cache_hooks()
            for key in list(cache.keys()):
                hook_type, _ = HookPoint.parse(key)
                cache[key] = self.adapter.format_cache_item(hook_type, cache[key])
            self.cache = cache._data

    @contextmanager
    def run_with_hooks(self, hooks: List[Tuple[str, Callable]]):
        """Context manager for patching activations.

        Args:
            hooks: List of (hook_name, hook_fn) tuples where:
                - hook_name: Full hook name like 'lm.blocks.5.hook_resid_post'
                  or with wildcard like 'lm.blocks.*.hook_resid_post'
                - hook_fn: Callable with signature (module, args, kwargs, output) -> output | None
                  Return modified output, or None to keep original.

        Valid hook points (post-hooks, no virtual hooks):
            - hook_resid_post: layer output
            - attn.hook_q, attn.hook_k, attn.hook_v: QKV projections
            - attn.hook_out: attention output
            - mlp.hook_pre, mlp.hook_pre_linear: gate/up projections
            - mlp.hook_out: MLP output

        Valid pre-hook points:
            - hook_resid_pre: layer input (residual stream before attention)
            - attn.hook_in: attention block input
            - attn.hook_mask: attention mask intervention (use with BlockAttention/SetAttentionMask)
            - attn.hook_z: pre o_proj, use with PatchHead for individual head patching
            - mlp.hook_in: MLP block input
            - mlp.hook_post: post-activation (after gate * up)

        Example:
            def scale_hook(module, args, kwargs, output):
                return output * 0.5

            with model.run_with_hooks([('lm.blocks.5.hook_resid_post', scale_hook)]):
                model.forward(inputs)

        Example with PatchHead (head-level patching):
            from vlm_spectra.core.patch_hooks import PatchHead

            num_heads = model.adapter.lm_num_heads
            head_dim = model.adapter.lm_head_dim
            replacement = torch.zeros(head_dim, device=model.device)

            patch = PatchHead(head_idx=0, replacement=replacement, num_heads=num_heads)
            with model.run_with_hooks([("lm.blocks.5.attn.hook_z", patch)]):
                model.forward(inputs)

        Yields:
            None
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
        input_ids = inputs["input_ids"]
        if input_ids.dim() > 1 and input_ids.size(0) > 1:
            raise ValueError(
                "Batch size > 1 not supported in get_image_token_range yet"
            )
        input_ids = input_ids.squeeze(0)
        image_token_id = self.adapter.get_image_token_id()

        image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

        if len(image_token_positions) == 0:
            raise ValueError("No image tokens found in the input sequence")

        start_index = image_token_positions[0].item()
        end_index = image_token_positions[-1].item()

        return start_index, end_index

    def generate_patch_overview(self, image: Image.Image, **kwargs) -> Image.Image:
        from vlm_spectra.visualization.patch_overview import (
            generate_patch_overview as _generate_patch_overview,
        )

        info = self.process_image(image)
        return _generate_patch_overview(info, **kwargs)

    def prepare_messages_batch(
        self,
        tasks: List[str],
        images: List[Image.Image],
        append_text: str = "",
        assistant_prefill: str | None = "",
        return_text: bool = False,
    ):
        return self._processor.prepare_inputs_batch(
            tasks=tasks,
            images=images,
            append_text=append_text,
            assistant_prefill=assistant_prefill,
            return_text=return_text,
        )

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
        needs_eager = kwargs.get("output_attentions", False)
        if needs_eager and hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        try:
            with torch.no_grad() if not require_grads else nullcontext():
                outputs = self.model.forward(
                    **inputs,
                    return_dict=return_dict,
                    **kwargs,
                )
        finally:
            if needs_eager and hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation(self.adapter._original_attn_impl)
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
        needs_eager = kwargs.get("output_attentions", False)
        if needs_eager and hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        try:
            with torch.no_grad() if not require_grads else nullcontext():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    return_dict_in_generate=return_dict_in_generate,
                    **kwargs,
                )
        finally:
            if needs_eager and hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation(self.adapter._original_attn_impl)

        return outputs
