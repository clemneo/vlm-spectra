# Plan: Unified Hook Interface

## Overview

Consolidate `run_with_hooks` and `run_with_module_hooks` into a single `run_with_hooks` interface where hooks declare their target via a `hook_point` class attribute.

## Files to Modify

1. `src/vlm_spectra/hooks/base.py` - Add `hook_point` to base class
2. `src/vlm_spectra/hooks/patch_hooks.py` - Add `hook_point` to each hook class, remove ZeroAblationHook and MeanAblationHook
3. `src/vlm_spectra/core/hooked_vlm.py` - Rewrite `run_with_hooks`, remove `run_with_module_hooks`
4. `tests/test_activation_patching.py` - Update tests to use unified interface

## Design

### Hook Point Naming Convention

Use the same naming as the caching system:

| hook_point | Target Module | Shape |
|------------|---------------|-------|
| `lm.layer.pre` | Decoder layer (pre) | `(batch, seq, hidden)` |
| `lm.layer.post` | Decoder layer (post) | `(batch, seq, hidden)` |
| `lm.mlp.out` | MLP module | `(batch, seq, hidden)` |
| `lm.attn.out` | Attention o_proj | `(batch, seq, hidden)` |
| `lm.attn.pre` | Attention (pre) | inputs to attention |

### Hook Base Class

```python
# src/vlm_spectra/hooks/base.py
class Hook(ABC):
    layer: int
    hook_point: str = "lm.layer.post"  # default

    @abstractmethod
    def __call__(self, module, args, kwargs, output):
        """All hooks receive (module, args, kwargs, output).

        For pre-hooks (hook_point ends with .pre), output is None.
        Return modified output, or None to keep original.
        """
        pass
```

### Production Hooks

```python
# src/vlm_spectra/hooks/patch_hooks.py
class PatchResidualHook(Hook):
    hook_point = "lm.layer.post"
    # ... rest unchanged

class PatchHeadHook(Hook):
    hook_point = "lm.attn.out"  # Now hooks the right module!
    # ... rest unchanged

class PatchMLPHook(Hook):
    hook_point = "lm.mlp.out"   # Now hooks the right module!
    # ... rest unchanged
```

Note: ZeroAblationHook and MeanAblationHook are removed.

### Unified run_with_hooks

```python
# src/vlm_spectra/core/hooked_vlm.py
@contextmanager
def run_with_hooks(self, hooks):
    handles = []

    for hook in hooks:
        module = self._get_module_for_hook_point(hook.hook_point, hook.layer)
        is_pre = hook.hook_point.endswith(".pre")

        if is_pre:
            handle = module.register_forward_pre_hook(
                self._wrap_pre_hook(hook), with_kwargs=True
            )
        else:
            handle = module.register_forward_hook(hook, with_kwargs=True)
        handles.append(handle)

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

def _get_module_for_hook_point(self, hook_point: str, layer: int):
    """Map hook_point string to actual module."""
    if hook_point in ("lm.layer.pre", "lm.layer.post"):
        return self.adapter.get_lm_layer(layer)
    elif hook_point == "lm.mlp.out":
        return self.adapter.get_lm_mlp(layer)
    elif hook_point in ("lm.attn.out", "lm.attn.pre"):
        return self.adapter.get_lm_attn(layer)
    else:
        raise ValueError(f"Unknown hook_point: {hook_point}")

def _wrap_pre_hook(self, hook):
    """Wrap hook to match pre-hook signature while keeping unified API."""
    def wrapper(module, args, kwargs):
        # Call hook with output=None for pre-hooks
        result = hook(module, args, kwargs, None)
        if result is not None:
            return result  # Modified args/kwargs
        return args, kwargs
    return wrapper
```

### Delete run_with_module_hooks

Remove `run_with_module_hooks` and `run_with_attn_hooks` entirely. The unified `run_with_hooks` handles all cases.

## Implementation Steps

### Step 1: Update Hook Base Class
- Add `hook_point: str = "lm.layer.post"` as class attribute
- Update docstring to explain the convention

### Step 2: Update Production Hooks
- `PatchResidualHook`: `hook_point = "lm.layer.post"` (default, explicit)
- `PatchHeadHook`: `hook_point = "lm.attn.out"`
- `PatchMLPHook`: `hook_point = "lm.mlp.out"`
- Remove `ZeroAblationHook` and `MeanAblationHook`

### Step 3: Rewrite run_with_hooks
- Add `_get_module_for_hook_point` method
- Add `_wrap_pre_hook` method for pre-hooks
- Rewrite `run_with_hooks` to dispatch based on `hook.hook_point`

### Step 4: Remove Old Context Managers
- Delete `run_with_module_hooks`
- Delete `run_with_attn_hooks`

### Step 5: Update Tests
- Remove test classes that used the old interface (TestAttnHooks, TestModuleHooks)
- Remove TestZeroAblationHook and TestMeanAblationHook
- Update TestPatchHeadHook and TestPatchMLPHook to use unified interface
- All tests should now use `run_with_hooks` exclusively

## Verification

```bash
uv run pytest tests/test_activation_patching.py -v --model uitars1.5-7b
```

Expected: All tests pass, including:
- `TestPatchHeadHook` tests (previously failing)
- `TestPatchMLPHook::test_patch_mlp_replaces_token` (previously failing)
