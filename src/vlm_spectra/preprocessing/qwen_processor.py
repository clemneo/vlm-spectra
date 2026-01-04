from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.preprocessing.base_processor import BaseProcessor
from vlm_spectra.preprocessing.utils.vision_info import process_vision_info


UI_TARS_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\\\', \\\\\", and \\\\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\\\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\\\', \\\", and \\\\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


class QwenProcessor(BaseProcessor):
    """Qwen2.5-VL preprocessing wrapper."""

    def __init__(self, hf_processor, default_prompt: Optional[str] = None) -> None:
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
            rendered_text = self.processor.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )
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

        if return_text:
            return inputs, rendered_text
        return inputs

    def _render_prompt(self, text: str, prompt_template: Optional[str]) -> str:
        template = prompt_template or self.default_prompt
        if not template:
            return text
        try:
            return template.format(language="English", instruction=text, text=text)
        except KeyError:
            return text
