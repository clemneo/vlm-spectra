from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from contextlib import nullcontext, contextmanager

from vlm_spectra.models.model_prompts import UI_TARS_PROMPT
from vlm_spectra.utils.qwen_25_vl_utils import process_vision_info, smart_resize, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS

SUPPORTED_QWEN_25_VL_MODELS = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]

SUPPORTED_MODELS = [
    *SUPPORTED_QWEN_25_VL_MODELS,
]


class HookedVLM:
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
            self.device = next(self.model.parameters()).device # Get the device of the first param
        else:
            self.device = device

    def generate(
        self,
        inputs,
        max_new_tokens: int = 512,
        output_hidden_states: bool = False,
        require_grads: bool = False,
    ):
        """Prepare inputs with:

        `inputs = self._prepare_messages(task, image)`
        """

        inputs = inputs.to("cuda" if self.device != "cpu" else "cpu")
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)
        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=output_hidden_states,
            )

        return outputs

    def forward(self, inputs, output_hidden_states: bool = False, require_grads: bool = False):
        inputs = inputs.to("cuda" if self.device != "cpu" else "cpu")
        if self.device != "auto":   
            inputs = inputs.to(self.device)
        self.model.eval()
        if require_grads:
            self.model.requires_grad_(True)
        with torch.no_grad() if not require_grads else nullcontext():
            outputs = self.model.forward(
                **inputs,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        return outputs

    def prepare_messages(self, task: str, image: Image, append_text: str = "", return_text: bool = False):
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

        # Preparation for inference
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
    def run_with_hooks(self, hooks, test=None):
        """Hard coded with layer hooks for now. TODO: make more general"""
        handles = []

        for hook in hooks:
            handle = self.model.language_model.layers[hook.layer].register_forward_hook(hook, with_kwargs=True)
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
            if hasattr(hook, 'layer') and hasattr(hook, 'module_name'):
                if hook.module_name == 'o_proj':
                    module = self.model.language_model.layers[hook.layer].self_attn.o_proj
                else:
                    raise ValueError(f"Unsupported module name: {hook.module_name}")
            elif hasattr(hook, 'module'):
                module = hook.module
            else:
                raise ValueError("Hook must have either (layer, module_name) or module attribute")
            
            # Register forward_pre_hook 
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
                'norm': self.model.language_model.norm,
                'lm_head': self.model.lm_head,
                'tokenizer': self.processor.tokenizer,
            }
        }
        return model_to_components[type(self.model)]

    def generate_patch_overview(self, image: Image, with_labels: bool = True, start_number: int = 1) -> Image:
        """Generate a patch overview visualization showing how the image is divided into patches"""
        
        # Get original image dimensions
        width, height = image.size
        
        # Calculate resized dimensions using the same logic as the model
        resized_height, resized_width = smart_resize(
            height, width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS
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
            draw.line([(0, y_start), (resized_width, y_start)], fill='red', width=1)
            # End of patch
            y_end = (i + 1) * effective_patch_size - 1
            draw.line([(0, y_end), (resized_width, y_end)], fill='red', width=1)
            
        for j in range(merged_grid_w):
            # Start of patch
            x_start = j * effective_patch_size
            draw.line([(x_start, 0), (x_start, resized_height)], fill='red', width=1)
            # End of patch
            x_end = (j + 1) * effective_patch_size - 1
            draw.line([(x_end, 0), (x_end, resized_height)], fill='red', width=1)
        
        # Calculate appropriate font size based on patch size
        font_size = max(14, min(32, effective_patch_size // 3))
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        if with_labels:
            # Add patch numbers - only show every 10th patch to reduce clutter
            patch_num = start_number
            for i in range(merged_grid_h):
                for j in range(merged_grid_w):
                # Only show numbers for patches divisible by 10, plus first and last
                    if patch_num % 10 == 1 or patch_num == merged_grid_h * merged_grid_w:
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
                        draw.rectangle([
                            x - 1, y - 1,
                            x + text_width + 1, y + text_height + 4
                        ], fill='white', outline='red')
                        
                        # Draw text
                        draw.text((x, y), text, fill='red', font=font)
                    
                    patch_num += 1
        
        return overlay_image