from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
from contextlib import nullcontext, contextmanager
from typing import Dict, Any, Tuple, Union

from vlm_spectra.models.model_prompts import UI_TARS_PROMPT
from vlm_spectra.utils.qwen_25_vl_utils import (
    process_vision_info,
    smart_resize,
    IMAGE_FACTOR,
    MIN_PIXELS,
    MAX_PIXELS,
)
from vlm_spectra.models.ModelAdapter import get_model_adapter

SUPPORTED_QWEN_25_VL_MODELS = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]

SUPPORTED_MODELS = [
    *SUPPORTED_QWEN_25_VL_MODELS,
]


class HookedVLM:
    """
    A wrapper around Vision-Language Models (VLMs) that provides interpretability hooks.

    This class wraps Hugging Face VLMs with interpretability functionality, allowing
    for hidden state extraction, attention pattern analysis, and logit lens visualization.
    Currently supports ByteDance-Seed/UI-TARS-1.5-7B (Qwen2.5-VL based).

    Args:
        model_name: Name of the model to load. Must be in SUPPORTED_MODELS.
        device: Device to load model on. "auto" uses the model's default device placement.
    """

    def __init__(
        self, model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B", device: str = "auto"
    ):
        assert model_name in SUPPORTED_MODELS, f"Model {model_name} not supported"
        self.model_name = model_name
        if model_name == "ByteDance-Seed/UI-TARS-1.5-7B":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=device
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.prompt = UI_TARS_PROMPT

        if device == "auto":
            self.device = next(
                self.model.parameters()
            ).device  # Get the device of the first param
        else:
            self.device = device

        self.adapter = get_model_adapter(self.model)
        self.adapter.set_processor(self.processor)
        self.cache = None

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 512,
        require_grads: bool = False,
        do_sample: bool = False,
        return_dict_in_generate: bool = True,
        **kwargs,
    ) -> Any:
        """
        Generate text from the model given prepared inputs.

        Args:
            inputs: Prepared model inputs from prepare_messages()
            max_new_tokens: Maximum number of new tokens to generate
            require_grads: Whether to enable gradients during generation
            do_sample: Whether to use sampling for generation
            return_dict_in_generate: Whether to return detailed generation info
            **kwargs: Additional arguments passed to model.generate()

        Returns:
            Generation outputs from the model
        """

        # inputs = inputs.to("cuda" if self.device != "cpu" else "cpu")
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
        self, inputs: Dict[str, torch.Tensor], require_grads: bool = False, return_dict: bool = True, **kwargs
    ) -> Any:
        """
        Run a forward pass through the model.

        Args:
            inputs: Prepared model inputs from prepare_messages()
            require_grads: Whether to enable gradients during forward pass
            return_dict: Whether to return outputs as a dictionary
            **kwargs: Additional arguments passed to model.forward()

        Returns:
            Forward pass outputs from the model
        """
        # inputs = inputs.to("cuda" if self.device != "cpu" else "cpu")
        inputs = inputs.to(self.device)
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)
        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.forward(
                **inputs,
                return_dict=True,
                **kwargs,
            )
        return outputs

    ## TODO: rewrite to be more general
    def prepare_messages(
        self,
        task: str,
        image: Image.Image,
        append_text: str = "",
        assistant_prefill: str = "",
        return_text: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], str]]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt.format(
                            language="English", instruction=task
                        ),
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                ],
            }
        ]

        # Add assistant prefill if provided
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        # Preparation for inference
        if assistant_prefill:
            # Use continue_final_message when we have assistant prefill
            text = self.processor.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )
        else:
            # Use add_generation_prompt when no prefill (current behavior)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        if append_text:
            text += append_text

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if return_text:
            return inputs, text
        else:
            return inputs

    @contextmanager
    def run_with_cache(self, cache_set: list):
        # keys could be: resid_pre, attn_out, resid_mid, mlp_out, resid_post
        handles = []

        # sanity checks
        for hook_pos in cache_set:
            if "lm" not in hook_pos:
                raise NotImplementedError("Only LM hooks are supported for now")

        cache = {}
        input_cache = {}

        def save_output_hook(layer, hook_pos):
            def hook(module, args, kwargs, output):
                cache[(hook_pos, layer)] = output

            return hook

        def save_input_hook(layer, hook_pos):
            def hook(module, args, kwargs, output):
                # For attention pattern computation, we need more than just hidden_states
                if hook_pos == "lm_attn_pattern":
                    # Capture all the arguments we need for accurate attention computation
                    hook_data = {}
                    if len(args) > 0:
                        hook_data["hidden_states"] = args[0]
                    elif "hidden_states" in kwargs:
                        hook_data["hidden_states"] = kwargs["hidden_states"]

                    # Capture additional arguments for accurate attention computation
                    hook_data["attention_mask"] = kwargs.get("attention_mask", None)
                    hook_data["position_ids"] = kwargs.get("position_ids", None)
                    hook_data["position_embeddings"] = kwargs.get(
                        "position_embeddings", None
                    )

                    # # Debug print to see what we're capturing
                    # print(f'Layer {layer} hook capture:')
                    # print(f'  attention_mask: {hook_data["attention_mask"] is not None}')
                    # print(f'  position_ids: {hook_data["position_ids"] is not None}')
                    # print(f'  position_embeddings: {hook_data["position_embeddings"] is not None}')
                    # if hook_data["position_embeddings"] is not None:
                    #     cos, sin = hook_data["position_embeddings"]
                    #     print(f'  cos shape: {cos.shape}, sin shape: {sin.shape}')

                    input_cache[(hook_pos, layer)] = hook_data
                else:
                    # Regular handling for other hook types
                    if len(args) > 0:
                        input_cache[(hook_pos, layer)] = args[0]
                    else:
                        # For modules called with keyword arguments, check kwargs
                        if "hidden_states" in kwargs:
                            input_cache[(hook_pos, layer)] = kwargs["hidden_states"]

            return hook

        # Something about registering forward hook and saving output
        lm_layer_list = list(range(self.adapter.lm_num_layers))

        for hook_pos in cache_set:
            if "lm" in hook_pos:
                if "resid_pre" in hook_pos or "resid_post" in hook_pos:
                    for layer in lm_layer_list:
                        handle = self.adapter.lm_layers[layer].register_forward_hook(
                            save_output_hook(layer, hook_pos), with_kwargs=True
                        )
                        handles.append(handle)
                elif "attn_out" in hook_pos:
                    for layer in lm_layer_list:
                        # Hook the input to o_proj to get concatenated heads before final projection
                        handle = self.adapter.lm_o_proj[layer].register_forward_hook(
                            save_input_hook(layer, hook_pos), with_kwargs=True
                        )
                        handles.append(handle)
                elif "mlp_out" in hook_pos:
                    for layer in lm_layer_list:
                        handle = self.adapter.lm_mlp[layer].register_forward_hook(
                            save_output_hook(layer, hook_pos), with_kwargs=True
                        )
                        handles.append(handle)
                elif "attn_pattern" in hook_pos:
                    for layer in lm_layer_list:
                        # Hook the attention module to capture hidden states input
                        handle = self.adapter.lm_attn[layer].register_forward_hook(
                            save_input_hook(layer, hook_pos), with_kwargs=True
                        )
                        handles.append(handle)
                elif "resid_mid" in hook_pos:
                    raise NotImplementedError(
                        "Resid_mid hooks are not supported for now"
                    )
            else:
                raise NotImplementedError("Only LM hooks are supported for now")

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

            # Process attention outputs to compute per-head contributions
            for hook_pos in cache_set:
                if "attn_out" in hook_pos:
                    for layer in lm_layer_list:
                        if (hook_pos, layer) in input_cache:
                            concatenated_heads = input_cache[(hook_pos, layer)]
                            per_head_contribs = (
                                self.adapter.compute_per_head_contributions(
                                    concatenated_heads, layer
                                )
                            )
                            cache[(hook_pos, layer)] = per_head_contribs
                elif "attn_pattern" in hook_pos:
                    for layer in lm_layer_list:
                        if (hook_pos, layer) in input_cache:
                            hook_data = input_cache[(hook_pos, layer)]
                            if isinstance(hook_data, dict):
                                # New format with additional parameters
                                attn_patterns = self.adapter.compute_attention_patterns(
                                    hidden_states=hook_data["hidden_states"],
                                    layer=layer,
                                    attention_mask=hook_data.get("attention_mask"),
                                    position_ids=hook_data.get("position_ids"),
                                    position_embeddings=hook_data.get(
                                        "position_embeddings"
                                    ),
                                )
                            else:
                                # Fallback for old format
                                attn_patterns = self.adapter.compute_attention_patterns(
                                    hook_data, layer
                                )
                            cache[(hook_pos, layer)] = attn_patterns

            cache = self.adapter.format_cache(
                cache
            )  # format each item to make it a single tensor

            self.cache = cache

    # TODO: Rewrite hooks to use the adapter
    @contextmanager
    def run_with_hooks(self, hooks, test=None):
        """Hard coded with layer hooks for now. TODO: make more general"""
        handles = []

        for hook in hooks:
            handle = self.model.language_model.layers[hook.layer].register_forward_hook(
                hook, with_kwargs=True
            )
            handles.append(handle)

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def run_with_attn_hooks(self, hooks):
        """Context manager for attention-level pre-hooks, specifically for o_proj layers."""
        handles = []
        for hook in hooks:
            handle = self.adapter.lm_attn[hook.layer].register_forward_pre_hook(
                hook, with_kwargs=True
            )
            handles.append(handle)
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def run_with_module_hooks(self, hooks):
        """Context manager for module-level pre-hooks, specifically for o_proj layers."""
        handles = []

        for hook in hooks:
            # Resolve module path based on hook configuration
            if hasattr(hook, "layer") and hasattr(hook, "module_name"):
                if hook.module_name == "o_proj":
                    module = self.model.language_model.layers[
                        hook.layer
                    ].self_attn.o_proj
                elif hook.module_name == "mlp":
                    module = self.model.language_model.layers[hook.layer].mlp
                elif hook.module_name == "self_attn":
                    module = self.model.language_model.layers[hook.layer].self_attn
                else:
                    raise ValueError(f"Unsupported module name: {hook.module_name}")
            elif hasattr(hook, "module"):
                module = hook.module
            else:
                raise ValueError(
                    "Hook must have either (layer, module_name) or module attribute"
                )

            # Register appropriate hook type
            if hasattr(hook, "hook_type") and hook.hook_type == "post":
                handle = module.register_forward_hook(hook, with_kwargs=False)
            else:
                # Default to pre-hook
                handle = module.register_forward_pre_hook(hook, with_kwargs=False)
            handles.append(handle)

        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def get_model_components(self):
        """Get model components needed for logit lens"""
        model_to_components = {
            Qwen2_5_VLForConditionalGeneration: {
                "norm": self.model.language_model.norm,
                "lm_head": self.model.lm_head,
                "tokenizer": self.processor.tokenizer,
            }
        }

        return model_to_components[type(self.model)]

    def generate_patch_overview(
        self, image: Image, with_labels: bool = True, start_number: int = 1
    ) -> Image:
        """Generate a patch overview visualization showing how the image is divided into patches"""

        # Get original image dimensions
        width, height = image.size

        # Calculate resized dimensions using the same logic as the model
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
        )

        # Get vision config for patch information
        vision_config = self.model.config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size

        # Calculate grid dimensions
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size

        # After spatial merge
        merged_grid_h = grid_h // spatial_merge_size
        merged_grid_w = grid_w // spatial_merge_size
        effective_patch_size = patch_size * spatial_merge_size

        # Resize the input image to the processed size
        resized_image = image.resize((resized_width, resized_height), Image.LANCZOS)

        # Create a copy to draw on
        overlay_image = resized_image.copy()
        draw = ImageDraw.Draw(overlay_image)

        # Draw lines at start and end of each patch
        for i in range(merged_grid_h):
            # Start of patch
            y_start = i * effective_patch_size
            draw.line([(0, y_start), (resized_width, y_start)], fill="red", width=1)
            # End of patch
            y_end = (i + 1) * effective_patch_size - 1
            draw.line([(0, y_end), (resized_width, y_end)], fill="red", width=1)

        for j in range(merged_grid_w):
            # Start of patch
            x_start = j * effective_patch_size
            draw.line([(x_start, 0), (x_start, resized_height)], fill="red", width=1)
            # End of patch
            x_end = (j + 1) * effective_patch_size - 1
            draw.line([(x_end, 0), (x_end, resized_height)], fill="red", width=1)

        # Calculate appropriate font size based on patch size
        font_size = max(14, min(32, effective_patch_size // 3))

        # Try to load a font, fall back to default if not available
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
            # Add patch numbers - only show every 10th patch to reduce clutter
            patch_num = start_number
            for i in range(merged_grid_h):
                for j in range(merged_grid_w):
                    # Only show numbers for patches divisible by 10, plus first and last
                    if (
                        patch_num % 10 == 1
                        or patch_num == merged_grid_h * merged_grid_w
                    ):
                        # Position text in center of patch area, accounting for grid line spacing
                        x = j * effective_patch_size + effective_patch_size // 4
                        y = i * effective_patch_size + effective_patch_size // 4

                        text = str(patch_num)
                        if font:
                            bbox = draw.textbbox((0, 0), text, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        else:
                            text_width, text_height = len(text) * 6, 10

                        # Draw small background rectangle in corner
                        draw.rectangle(
                            [x - 1, y - 1, x + text_width + 1, y + text_height + 4],
                            fill="white",
                            outline="red",
                        )

                        # Draw text
                        draw.text((x, y), text, fill="red", font=font)

                    patch_num += 1

        return overlay_image

    def get_image_token_range(self, inputs) -> tuple[int, int]:
        """
        Get the start and end indices of image tokens in the input sequence

        Args:
            inputs: The processed inputs from prepare_messages()

        Returns:
            Tuple of (start_index, end_index) where both indices are inclusive

        Raises:
            ValueError: If no image tokens are found in the sequence
        """
        # Extract input_ids and get image token ID from adapter
        input_ids = inputs["input_ids"].squeeze(0)
        image_token_id = self.adapter.get_image_token_id()

        # Find all positions with image tokens
        image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

        if len(image_token_positions) == 0:
            raise ValueError("No image tokens found in the input sequence")

        # Return first and last positions (both inclusive)
        start_index = image_token_positions[0].item()
        end_index = image_token_positions[-1].item()

        return start_index, end_index

    def prepare_messages_batch(
        self,
        tasks: list[str],
        images: list[Image],
        append_text: str = "",
        assistant_prefill: str = "",
        return_text: bool = False,
    ):
        """
        Prepare a batch of messages for processing multiple task-image pairs at once.

        Args:
            tasks: List of task strings
            images: List of PIL Images
            append_text: Text to append to each message
            assistant_prefill: Assistant prefill text for each message
            return_text: Whether to return the processed text along with inputs

        Returns:
            Batched inputs ready for model processing, optionally with texts

        Note:
            Batch processing provides improved throughput but may produce slightly
            different results compared to individual processing due to padding
            effects and attention mask interactions. Single-item batches should
            produce identical results to individual calls.
        """
        if len(tasks) != len(images):
            raise ValueError(
                f"Number of tasks ({len(tasks)}) must match number of images ({len(images)})"
            )

        # Prepare each message individually first
        batch_messages = []
        batch_texts = []

        for task, image in zip(tasks, images):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt.format(
                                language="English", instruction=task
                            ),
                        },
                        {
                            "type": "image",
                            "image": image,
                        },
                    ],
                }
            ]

            # Add assistant prefill if provided
            if assistant_prefill:
                messages.append({"role": "assistant", "content": assistant_prefill})

            # Preparation for inference
            if assistant_prefill:
                # Use continue_final_message when we have assistant prefill
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, continue_final_message=True
                )
            else:
                # Use add_generation_prompt when no prefill
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            if append_text:
                text += append_text

            batch_messages.append(messages)
            batch_texts.append(text)

        # Process vision info for all messages
        all_image_inputs = []
        all_video_inputs = []

        for messages in batch_messages:
            image_inputs, video_inputs = process_vision_info(messages)
            if image_inputs:
                all_image_inputs.extend(image_inputs)
            if video_inputs:
                all_video_inputs.extend(video_inputs)

        # Use processor to handle the batch
        inputs = self.processor(
            text=batch_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=all_video_inputs if all_video_inputs else None,
            padding=True,
            return_tensors="pt",
        )

        if return_text:
            return inputs, batch_texts
        else:
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
        """
        Forward pass for a batch of inputs.

        Args:
            inputs_list: List of prepared inputs (if None, will prepare from tasks/images)
            tasks: List of task strings (required if inputs_list is None)
            images: List of PIL Images (required if inputs_list is None)
            require_grads: Whether to compute gradients
            return_dict: Whether to return dict format
            **kwargs: Additional arguments for forward pass

        Returns:
            Batched model outputs
        """
        # Prepare inputs if not provided
        if inputs_list is None:
            if tasks is None or images is None:
                raise ValueError(
                    "Either inputs_list or both tasks and images must be provided"
                )
            inputs = self.prepare_messages_batch(tasks, images)
        else:
            # If inputs_list is provided, we need to collate them properly
            # For now, assume inputs_list contains already prepared inputs
            # This is a simplification - in practice you might need more complex collation
            if isinstance(inputs_list, list) and len(inputs_list) == 1:
                inputs = inputs_list[0]
            else:
                raise NotImplementedError(
                    "Complex collation of input_list not yet implemented. Use tasks/images instead."
                )

        # Move to device and run forward pass
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
        """
        Generate text for a batch of inputs.

        Args:
            inputs_list: List of prepared inputs (if None, will prepare from tasks/images)
            tasks: List of task strings (required if inputs_list is None)
            images: List of PIL Images (required if inputs_list is None)
            max_new_tokens: Maximum number of new tokens to generate
            require_grads: Whether to compute gradients
            do_sample: Whether to use sampling
            return_dict_in_generate: Whether to return dict format
            **kwargs: Additional arguments for generation

        Returns:
            Batched generation outputs
        """
        # Prepare inputs if not provided
        if inputs_list is None:
            if tasks is None or images is None:
                raise ValueError(
                    "Either inputs_list or both tasks and images must be provided"
                )
            inputs = self.prepare_messages_batch(tasks, images)
        else:
            # If inputs_list is provided, we need to collate them properly
            # For now, assume inputs_list contains already prepared inputs
            # This is a simplification - in practice you might need more complex collation
            if isinstance(inputs_list, list) and len(inputs_list) == 1:
                inputs = inputs_list[0]
            else:
                raise NotImplementedError(
                    "Complex collation of input_list not yet implemented. Use tasks/images instead."
                )

        # Move to device and run generation
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
