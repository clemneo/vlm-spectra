from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

from vlm_spectra.models.model_prompts import UI_TARS_PROMPT
from vlm_spectra.utils.qwen_25_vl_utils import process_vision_info

SUPPORTED_QWEN_25_VL_MODELS = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]

SUPPORTED_MODELS = [
    *SUPPORTED_QWEN_25_VL_MODELS,
]

class HookedVLM:
    def __init__(self, model_name: str = "ByteDance-Seed/UI-TARS-1.5-7B", device: str = "cuda"):
        assert model_name in SUPPORTED_MODELS, f"Model {model_name} not supported"
        self.model_name = model_name
        self.device = device
        if model_name == "ByteDance-Seed/UI-TARS-1.5-7B":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.prompt = UI_TARS_PROMPT
        

    def generate(self, task: str, image: Image, max_new_tokens: int = 512, output_hidden_states: bool = False):
        inputs = self._prepare_messages(task, image)
        inputs = inputs.to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=output_hidden_states
            )

        return outputs

    def _prepare_messages(self, task: str, image: Image):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt.format(language="English", instruction=task)},
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

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return inputs

