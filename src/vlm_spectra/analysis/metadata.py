from __future__ import annotations

from typing import Any, Dict

from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from vlm_spectra.preprocessing.utils.vision_info import (
    MAX_PIXELS,
    MIN_PIXELS,
    resolve_patch_params,
    smart_resize,
)


class VLMMetadataExtractor:
    """Extract metadata needed for logit lens visualization from various VLMs."""

    @staticmethod
    def extract_metadata_qwen(
        model: Qwen2_5_VLForConditionalGeneration,
        processor: AutoProcessor,
        inputs: Dict[str, Any],
        original_image: Image.Image,
    ) -> Dict[str, Any]:
        """Extract metadata from Qwen model inputs/outputs."""
        width, height = original_image.size

        vision_config = model.config.vision_config
        model_name = getattr(model.config, "name_or_path", None)
        patch_size, spatial_merge_size = resolve_patch_params(
            vision_config, model_name
        )
        resize_factor = patch_size * spatial_merge_size

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=resize_factor,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        merged_grid_h = grid_h // spatial_merge_size
        merged_grid_w = grid_w // spatial_merge_size

        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        token_labels = []
        image_token_positions = []
        img_token_counter = 0

        for token_id in input_ids.tolist():
            if token_id == image_token_id:
                token_labels.append(f"<IMG{(img_token_counter + 1):03d}>")
                image_token_positions.append(len(token_labels) - 1)
                img_token_counter += 1
            else:
                token_labels.append(processor.tokenizer.decode([token_id]))

        return {
            "token_labels": token_labels,
            "image_token_positions": image_token_positions,
            "image_size": (resized_width, resized_height),
            "grid_size": (merged_grid_h, merged_grid_w),
            "patch_size": patch_size * spatial_merge_size,
            "num_image_tokens": len(image_token_positions),
            "total_patches": merged_grid_h * merged_grid_w,
        }

    @staticmethod
    def extract_metadata_llava(
        model: Any,
        processor: Any,
        inputs: Dict[str, Any],
        original_image: Image.Image,
        image_size: int = 336,
        patch_size: int = 14,
    ) -> Dict[str, Any]:
        """Extract metadata from LLaVA model inputs/outputs."""
        _ = model
        _ = original_image
        resized_size = (image_size, image_size)
        grid_size = image_size // patch_size

        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

        token_labels = []
        image_token_positions = []

        for token_id in input_ids.tolist():
            if token_id == image_token_id:
                num_patches = grid_size * grid_size
                for j in range(num_patches):
                    token_labels.append(f"<IMG{(j + 1):03d}>")
                    image_token_positions.append(len(token_labels) - 1)
            else:
                token_labels.append(processor.tokenizer.decode([token_id]))

        return {
            "token_labels": token_labels,
            "image_token_positions": image_token_positions,
            "image_size": resized_size,
            "grid_size": (grid_size, grid_size),
            "patch_size": patch_size,
            "num_image_tokens": len(image_token_positions),
            "total_patches": grid_size * grid_size,
        }
