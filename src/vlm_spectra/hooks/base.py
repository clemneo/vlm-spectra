from __future__ import annotations

from abc import ABC, abstractmethod


class Hook(ABC):
    """Base class for all hooks.

    Hooks declare their target via the `hook_point` class attribute:
    - "lm.layer.pre": Decoder layer (pre-hook)
    - "lm.layer.post": Decoder layer (post-hook)
    - "lm.mlp.out": MLP module output
    - "lm.attn.out": Attention o_proj output
    - "lm.attn.pre": Attention (pre-hook)
    """

    layer: int
    hook_point: str = "lm.layer.post"

    @abstractmethod
    def __call__(self, module, args, kwargs, output):
        """Apply the hook to a module call.

        For pre-hooks (hook_point ends with .pre), output is None.
        Return modified output, or None to keep original.
        """
