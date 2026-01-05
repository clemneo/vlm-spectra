from __future__ import annotations

import json
from typing import Callable, Optional, Union


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
type(content='xxx') # Use escape characters \\\\', \\\\", and \\\\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\\\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\\\', \\\", and \\\\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

TOOL_CALL_TEMPLATE = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_descs}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
Return coordinates as absolute pixels on the provided screenshot (after any resizing you apply)."""


def _qwen3_vl_tool_schema(display_width: int, display_height: int) -> dict[str, object]:
    return {
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
            "You must click on desktop icons to start applications.\n"
            "* Some applications may take time to start or process actions, so you may need to wait and take "
            "successive screenshots to see the results of your actions.\n"
            f"* The screen's resolution is {display_width}x{display_height}.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
            "screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting "
            "your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
            "Don't click boxes on their edges.\n"
        ),
        "parameters": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform.",
                    "enum": [
                        "key",
                        "type",
                        "mouse_move",
                        "left_click",
                        "left_click_drag",
                        "right_click",
                        "middle_click",
                        "double_click",
                        "triple_click",
                        "scroll",
                        "hscroll",
                        "wait",
                        "terminate",
                        "answer",
                    ],
                },
                "keys": {"type": "array", "description": "Required only by action=key."},
                "text": {"type": "string", "description": "Required only by action=type and action=answer."},
                "coordinate": {
                    "type": "array",
                    "description": "(x, y) pixel coordinates to move/click/drag.",
                },
                "pixels": {
                    "type": "number",
                    "description": "Scroll amount. Positive scrolls up, negative scrolls down.",
                },
                "time": {"type": "number", "description": "Seconds to wait (action=wait)."},
                "status": {
                    "type": "string",
                    "enum": ["success", "failure"],
                    "description": "Status (action=terminate).",
                },
            },
        },
    }


def build_qwen3_vl_prompt(task: str, display_width: int = 1000, display_height: int = 1000) -> str:
    tool_schema = _qwen3_vl_tool_schema(display_width, display_height)
    tool_json = json.dumps({"type": "function", "function": tool_schema})
    return TOOL_CALL_TEMPLATE.format(tool_descs=tool_json) + f"\n\nUser task: {task}"


DefaultPrompt = Optional[Union[str, Callable[[str], str]]]


def default_prompt_for_model(model_name: str) -> DefaultPrompt:
    if model_name == "ByteDance-Seed/UI-TARS-1.5-7B":
        return UI_TARS_PROMPT
    if model_name == "Qwen/Qwen3-VL-8B-Instruct":
        return build_qwen3_vl_prompt
    return None
