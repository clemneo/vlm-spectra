# vlm-spectra Rewrite Plan

## Overview

Restructure vlm-spectra from a monolithic, UI-TARS-specific library into a modular, general-purpose VLM mechanistic interpretability toolkit. This phase focuses on **refactoring the existing Qwen/UI-TARS code** into a clean architecture that can later support additional models.

**Key Goals:**
- Decompose monolithic HookedVLM (768 lines) into single-responsibility components
- Create extensible architecture for future multi-model support
- TransformerLens-style explicit API
- Remove GUI-agent-specific hardcoding
- All existing tests must pass after refactor
- Exclude web app from rewrite

**Scope for this session:**
- Restructure existing Qwen2.5-VL/UI-TARS support only
- No new model implementations (LLaVA, Molmo, etc.) - that's a future session

---

## New Directory Structure

```
vlm-spectra/src/vlm_spectra/
├── __init__.py                      # Public API exports
├── core/
│   ├── __init__.py
│   ├── hooked_vlm.py               # Thin orchestrator (~200 lines)
│   ├── activation_cache.py          # Intuitive cache API
│   └── hook_manager.py              # Hook registration/lifecycle
├── models/
│   ├── __init__.py
│   ├── registry.py                  # Model factory with decorator registration
│   ├── base_adapter.py              # Abstract ModelAdapter interface
│   └── adapters/
│       ├── __init__.py
│       └── qwen25_vl.py            # Qwen2.5-VL adapter (from ModelAdapter.py)
├── preprocessing/
│   ├── __init__.py
│   ├── base_processor.py            # Abstract processor interface
│   ├── qwen_processor.py            # Qwen preprocessing (from HookedVLM.prepare_messages)
│   └── utils/
│       ├── __init__.py
│       └── vision_info.py           # From qwen_25_vl_utils.py
├── hooks/
│   ├── __init__.py
│   ├── base.py                      # Base hook classes
│   ├── capture_hooks.py             # Activation capture
│   └── patch_hooks.py               # Activation patching
├── analysis/
│   ├── __init__.py
│   ├── logit_lens.py                # Logit lens computation
│   └── metadata.py                  # Token/patch metadata (from vlm_metadata.py)
├── visualization/
│   ├── __init__.py
│   └── logit_lens_html.py           # HTML generation (from create_logit_lens.py)
├── utils/                           # Keep existing, move qwen utils
│   └── __init__.py
└── web_app/                         # UNCHANGED - not in scope
```

---

## Key Components

### 1. HookedVLM (core/hooked_vlm.py)

Thin orchestrator composing adapter, hook manager, and cache.

```python
class HookedVLM:
    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "auto") -> "HookedVLM"

    def forward(self, inputs, **kwargs) -> ModelOutput
    def generate(self, inputs, max_new_tokens=512, **kwargs) -> GenerateOutput
    def prepare_inputs(self, text: str, image: Image, **kwargs) -> Dict

    @contextmanager
    def run_with_cache(self, cache_names: List[str]) -> ActivationCache

    @contextmanager
    def run_with_hooks(self, hooks: List[Hook])
```

### 2. ActivationCache (core/activation_cache.py)

Intuitive activation storage with multiple access patterns.

```python
# Access patterns
cache["lm.attn.out", 5]           # Tuple key
cache.get_all_layers("lm.attn.out")  # Dict[int, Tensor]
cache.stack_layers("lm.attn.out")    # Stacked tensor
```

### 3. ModelAdapter (models/base_adapter.py)

Abstract interface for model-specific internals.

```python
class ModelAdapter(ABC):
    # Properties
    lm_num_layers, lm_num_heads, lm_hidden_dim, lm_head_dim
    vision_num_layers, vision_patch_size

    # Module accessors
    get_lm_layer(layer_idx), get_lm_attn(layer_idx), get_lm_mlp(layer_idx)
    get_lm_norm(), get_lm_head(), get_vision_layer(layer_idx)

    # Token info
    get_image_token_id(), get_image_token_range(input_ids)

    # Computation
    compute_per_head_contributions(...), compute_attention_patterns(...)
```

### 4. ModelRegistry (models/registry.py)

Decorator-based model registration.

```python
@ModelRegistry.register("qwen25-vl")
class Qwen25VLAdapter(ModelAdapter):
    SUPPORTED_MODELS = ["Qwen/Qwen2.5-VL-7B", "ByteDance-Seed/UI-TARS-1.5-7B", ...]

# Usage
model = HookedVLM.from_pretrained("Qwen/Qwen2.5-VL-7B")
```

---

## Hook Naming Convention

```
<component>.<subcomponent>.<position>

Examples:
- lm.layer.pre      # Residual stream input to layer
- lm.layer.post     # Residual stream output from layer
- lm.attn.out       # Per-head attention contributions
- lm.attn.pattern   # Attention weights
- lm.mlp.out        # MLP contribution to residual
- vision.layer.out  # Vision encoder outputs
```

**Cache Output Shapes:**
| Hook Name | Shape |
|-----------|-------|
| `lm.layer.pre/post` | `[batch, seq, hidden]` |
| `lm.attn.out` | `[batch, seq, heads, hidden]` |
| `lm.attn.pattern` | `[batch, heads, seq, seq]` |
| `lm.mlp.out` | `[batch, seq, hidden]` |

---

## Built-in Hooks Library

**Capture hooks** (hooks/capture_hooks.py):
- `CaptureOutputHook` - Capture module output
- `CaptureInputHook` - Capture module input

**Patch hooks** (hooks/patch_hooks.py):
- `PatchResidualHook(layer, token_idx, replacement)` - Replace residual at position
- `PatchHeadHook(layer, head_idx, replacement, token_idx)` - Replace head output
- `PatchMLPHook(layer, replacement, token_idx)` - Replace MLP output

**Ablation hooks** (hooks/ablation_hooks.py):
- `ZeroAblationHook(layer, positions)` - Zero out activations
- `MeanAblationHook(layer, positions)` - Replace with mean

---

## API Migration

```python
# BEFORE (current)
from vlm_spectra.models.HookedVLM import HookedVLM
model = HookedVLM(model_name="ByteDance-Seed/UI-TARS-1.5-7B")
inputs = model.prepare_messages(task, image)
with model.run_with_cache(["lm_attn_out"]):
    _ = model.forward(inputs)
attn = model.cache[("lm_attn_out", 5)]

# AFTER (new)
from vlm_spectra import HookedVLM
model = HookedVLM.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")
inputs = model.prepare_inputs(task, image)
with model.run_with_cache(["lm.attn.out"]) as cache:
    _ = model.forward(inputs)
attn = cache["lm.attn.out", 5]
```

---

## Testing Strategy

### Test Structure
```
tests/
├── unit/                    # No GPU, fast
│   ├── test_activation_cache.py
│   ├── test_hook_manager.py
│   └── test_preprocessing.py
├── integration/             # GPU required
│   ├── test_qwen_adapter.py
│   ├── test_llava_adapter.py
│   └── test_hooked_vlm_qwen.py
├── e2e/                     # Full workflows
│   ├── test_run_with_cache.py
│   └── test_patching_experiments.py
└── fixtures/conftest.py     # Shared fixtures
```

### Test Commands
```bash
uv run pytest                           # All tests
uv run pytest tests/unit/               # Unit only (fast)
uv run pytest tests/integration/ -v     # Integration with GPU
uv run pytest --cov=src/vlm_spectra     # With coverage
```

---

## Implementation Phases (Detailed)

### Phase 1: Core Infrastructure - ActivationCache

**Goal:** Create the `ActivationCache` class that will replace `model.cache` dictionary.

**Create `src/vlm_spectra/core/__init__.py`:**
```python
from .activation_cache import ActivationCache
from .hook_manager import HookManager
from .hooked_vlm import HookedVLM
```

**Create `src/vlm_spectra/core/activation_cache.py`:**
```python
class ActivationCache:
    """Cache for model activations with intuitive access."""

    def __init__(self):
        self._data: Dict[Tuple[str, int], torch.Tensor] = {}

    def __getitem__(self, key: Tuple[str, int]) -> torch.Tensor:
        """cache["lm.attn.out", 5] style access"""
        return self._data[key]

    def __setitem__(self, key: Tuple[str, int], value: torch.Tensor):
        self._data[key] = value

    def __contains__(self, key) -> bool:
        return key in self._data

    def keys(self) -> List[Tuple[str, int]]:
        return list(self._data.keys())

    def items(self):
        return self._data.items()

    def get_all_layers(self, name: str) -> Dict[int, torch.Tensor]:
        """Get all layers for a hook name as dict."""
        return {layer: tensor for (n, layer), tensor in self._data.items() if n == name}

    def stack_layers(self, name: str) -> torch.Tensor:
        """Stack all layers for a hook name into [num_layers, ...]."""
        layers = self.get_all_layers(name)
        return torch.stack([layers[i] for i in sorted(layers.keys())])

    def clear(self):
        self._data.clear()

    def detach(self):
        """Detach all tensors from computation graph."""
        for key in self._data:
            self._data[key] = self._data[key].detach()
```

**Create `tests/unit/test_activation_cache.py`:**
- Test `__getitem__`, `__setitem__`, `__contains__`
- Test `get_all_layers` returns correct subset
- Test `stack_layers` shapes correctly
- Test `clear` and `detach`

**Run:** `uv run pytest tests/unit/test_activation_cache.py`

---

### Phase 2: Core Infrastructure - HookManager

**Goal:** Extract hook registration logic from `HookedVLM.run_with_cache()` (lines 190-331).

**Create `src/vlm_spectra/core/hook_manager.py`:**
```python
class HookManager:
    """Manages PyTorch hook lifecycle for interpretability."""

    def __init__(self, adapter: "ModelAdapter"):
        self._adapter = adapter
        self._handles: List[RemovableHandle] = []

    def register_cache_hooks(self, cache: ActivationCache, names: List[str]):
        """Register hooks that capture activations into cache."""
        # Port logic from HookedVLM.run_with_cache lines 251-284
        # Maps hook names to modules and registers appropriate hooks

    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _get_module_for_hook(self, hook_name: str, layer: int) -> nn.Module:
        """Resolve hook name to actual module."""
        # "lm.attn.out" -> adapter.get_lm_o_proj(layer)
        # "lm.layer.post" -> adapter.get_lm_layer(layer)
        # etc.
```

**Key logic to extract from HookedVLM.run_with_cache:**
- `save_output_hook` function (line 203-207)
- `save_input_hook` function (line 209-246)
- Hook registration loop (lines 251-284)
- Cache formatting call to adapter (line 326-328)

**Create `tests/unit/test_hook_manager.py`:**
- Test hook registration with mock adapter
- Test hook removal cleans up handles
- Test module resolution for different hook names

**Run:** `uv run pytest tests/unit/test_hook_manager.py`

---

### Phase 3: Model Adapter Refactoring

**Goal:** Refactor existing `ModelAdapter.py` into proper base class and Qwen implementation.

**Create `src/vlm_spectra/models/base_adapter.py`:**
- Move abstract `ModelAdapter` class from current `ModelAdapter.py` (lines 49-133)
- Add new abstract methods:
  - `get_lm_layer(layer_idx) -> nn.Module`
  - `get_lm_attn(layer_idx) -> nn.Module`
  - `get_lm_o_proj(layer_idx) -> nn.Module`
  - `get_lm_mlp(layer_idx) -> nn.Module`
- Keep existing properties: `lm_num_layers`, `lm_num_heads`, `lm_hidden_dim`

**Create `src/vlm_spectra/models/adapters/__init__.py`:**
```python
from .qwen25_vl import Qwen25VLAdapter
```

**Create `src/vlm_spectra/models/adapters/qwen25_vl.py`:**
- Move `Qwen2_5_VLModelAdapter` from current `ModelAdapter.py` (lines 142-354)
- Rename to `Qwen25VLAdapter`
- Add `SUPPORTED_MODELS` class variable:
  ```python
  SUPPORTED_MODELS = [
      "ByteDance-Seed/UI-TARS-1.5-7B",
      "Qwen/Qwen2.5-VL-7B",
      "Qwen/Qwen2.5-VL-7B-Instruct",
  ]
  ```
- Implement new abstract methods by wrapping existing properties

**Create `src/vlm_spectra/models/registry.py`:**
```python
class ModelRegistry:
    _adapters: Dict[str, Type[ModelAdapter]] = {}
    _model_to_adapter: Dict[str, str] = {}

    @classmethod
    def register(cls, adapter_type: str):
        """Decorator to register adapter class."""
        def decorator(adapter_cls):
            cls._adapters[adapter_type] = adapter_cls
            for model_name in adapter_cls.SUPPORTED_MODELS:
                cls._model_to_adapter[model_name] = adapter_type
            return adapter_cls
        return decorator

    @classmethod
    def get_adapter_class(cls, model_name: str) -> Type[ModelAdapter]:
        adapter_type = cls._model_to_adapter.get(model_name)
        if not adapter_type:
            raise ValueError(f"Model {model_name} not supported")
        return cls._adapters[adapter_type]

    @classmethod
    def load(cls, model_name: str, device: str = "auto"):
        """Load model, processor, and create adapter."""
        # Load model using HF
        # Create appropriate adapter
        # Return (model, processor, adapter)
```

**Update Qwen adapter to use registry:**
```python
@ModelRegistry.register("qwen25-vl")
class Qwen25VLAdapter(ModelAdapter):
    ...
```

**Run:** `uv run pytest tests/unit/` (ensure no regressions)

---

### Phase 4: Preprocessing Decoupling

**Goal:** Extract `prepare_messages` logic from `HookedVLM` into separate processor.

**Create `src/vlm_spectra/preprocessing/base_processor.py`:**
```python
class BaseProcessor(ABC):
    """Abstract base for model-specific preprocessing."""

    @abstractmethod
    def prepare_inputs(
        self,
        text: str,
        image: Image.Image,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Convert text + image to model inputs."""
        pass
```

**Create `src/vlm_spectra/preprocessing/qwen_processor.py`:**
- Extract `prepare_messages` logic from `HookedVLM` (lines 131-188)
- Remove UI-TARS-specific prompt handling - make prompt a parameter
- Keep `assistant_prefill` and `append_text` logic

```python
class QwenProcessor(BaseProcessor):
    def __init__(self, hf_processor, default_prompt: str = None):
        self.processor = hf_processor
        self.default_prompt = default_prompt

    def prepare_inputs(
        self,
        text: str,
        image: Image.Image,
        prompt_template: str = None,
        assistant_prefill: str = "",
        return_text: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Port logic from HookedVLM.prepare_messages
        ...
```

**Move `src/vlm_spectra/utils/qwen_25_vl_utils.py` to `src/vlm_spectra/preprocessing/utils/vision_info.py`:**
- Keep `process_vision_info`, `smart_resize`, constants
- Update imports throughout codebase

**Run:** `uv run pytest` (ensure all tests still pass)

---

### Phase 5: New HookedVLM Implementation

**Goal:** Create new thin `HookedVLM` that composes the new components.

**Create `src/vlm_spectra/core/hooked_vlm.py`:**
```python
class HookedVLM:
    """Main entry point for VLM interpretability."""

    def __init__(self, model, processor, adapter, device="auto"):
        self.model = model
        self._processor = processor  # QwenProcessor instance
        self.adapter = adapter
        self._hook_manager = HookManager(adapter)
        self.device = device
        self.cache = None  # For backward compatibility

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "auto", **kwargs):
        """Factory method using registry."""
        model, hf_processor, adapter = ModelRegistry.load(model_name, device, **kwargs)
        processor = QwenProcessor(hf_processor)  # Will be generalized later
        return cls(model, processor, adapter, device)

    # Core methods - port from existing HookedVLM
    def forward(self, inputs, **kwargs) -> Any:
        # Port from lines 102-128

    def generate(self, inputs, max_new_tokens=512, **kwargs) -> Any:
        # Port from lines 62-100

    # Preprocessing - delegate to processor
    def prepare_inputs(self, text: str, image: Image.Image, **kwargs):
        return self._processor.prepare_inputs(text, image, **kwargs)

    # Backward compatibility alias
    def prepare_messages(self, task: str, image: Image.Image, **kwargs):
        return self.prepare_inputs(task, image, **kwargs)

    # Hook context managers
    @contextmanager
    def run_with_cache(self, cache_names: List[str]):
        """Cache activations during forward pass."""
        cache = ActivationCache()
        self._hook_manager.register_cache_hooks(cache, cache_names)
        try:
            yield cache
        finally:
            self._hook_manager.remove_all_hooks()
            # Format cache items using adapter
            for key in list(cache.keys()):
                cache[key] = self.adapter.format_cache_item(key[0], cache[key])
            self.cache = cache._data  # Backward compat

    @contextmanager
    def run_with_hooks(self, hooks):
        # Port from lines 333-348

    @contextmanager
    def run_with_attn_hooks(self, hooks):
        # Port from lines 350-379

    @contextmanager
    def run_with_module_hooks(self, hooks):
        # Port from lines 381-403

    # Properties delegated to adapter
    @property
    def lm_num_layers(self) -> int:
        return self.adapter.lm_num_layers

    @property
    def processor(self):
        """Access underlying HF processor for backward compat."""
        return self._processor.processor

    # Utility methods - port remaining
    def get_model_components(self):
        # Port from lines 405-420

    def get_image_token_range(self, inputs):
        # Port from lines 422-450
```

**Update `src/vlm_spectra/__init__.py`:**
```python
from .core.hooked_vlm import HookedVLM
from .core.activation_cache import ActivationCache
from .models.registry import ModelRegistry

# Backward compat - deprecated
from .models.HookedVLM import HookedVLM as LegacyHookedVLM
```

**Run all existing tests:** `uv run pytest`
- `tests/test_run_with_cache.py` - Must pass (validates cache correctness)
- `tests/hookedvlm_test.py` - Must pass
- `tests/test_batching.py` - Must pass
- `tests/test_prefill_functionality.py` - Must pass

---

### Phase 6: Hook Library

**Goal:** Create reusable hook classes from patterns in computer-interp experiments.

**Create `src/vlm_spectra/hooks/base.py`:**
```python
class Hook(ABC):
    """Base class for all hooks."""
    layer: int

    @abstractmethod
    def __call__(self, module, args, kwargs, output):
        pass
```

**Create `src/vlm_spectra/hooks/capture_hooks.py`:**
```python
class CaptureOutputHook(Hook):
    """Capture module output."""
    def __init__(self, cache: ActivationCache, hook_name: str, layer: int):
        self.cache = cache
        self.hook_name = hook_name
        self.layer = layer

    def __call__(self, module, args, kwargs, output):
        self.cache[self.hook_name, self.layer] = output
        return output

class CaptureInputHook(Hook):
    """Capture module input (hidden_states from args/kwargs)."""
    ...
```

**Create `src/vlm_spectra/hooks/patch_hooks.py`:**
Extract patterns from `computer-interp/scripts/square_patching_exp.py`:

```python
class PatchResidualHook(Hook):
    """Replace residual stream at specific token position."""
    def __init__(self, layer: int, token_idx: int, replacement: torch.Tensor):
        self.layer = layer
        self.token_idx = token_idx
        self.replacement = replacement

    def __call__(self, module, args, kwargs, output):
        output[0][0, self.token_idx] = self.replacement
        return output

class PatchHeadHook(Hook):
    """Replace attention head output."""
    def __init__(self, layer: int, head_idx: int, replacement: torch.Tensor,
                 token_idx: Optional[int] = None):
        ...

class ZeroAblationHook(Hook):
    """Zero out activations at specified positions."""
    ...
```

**Create `src/vlm_spectra/hooks/__init__.py`:**
```python
from .base import Hook
from .capture_hooks import CaptureOutputHook, CaptureInputHook
from .patch_hooks import PatchResidualHook, PatchHeadHook, ZeroAblationHook
```

**Run:** `uv run pytest`

---

### Phase 7: Analysis Module

**Goal:** Refactor analysis utilities from scattered locations.

**Create `src/vlm_spectra/analysis/metadata.py`:**
- Move `VLMMetadataExtractor` from `vlm_metadata.py`
- Keep `extract_metadata_qwen` method

**Create `src/vlm_spectra/analysis/logit_lens.py`:**
- Extract logit lens computation logic from `create_logit_lens.py`
- Separate computation from HTML generation:

```python
def compute_logit_lens(
    hidden_states: Dict[int, torch.Tensor],  # From cache
    norm: nn.Module,
    lm_head: nn.Module,
    tokenizer,
) -> Dict[int, torch.Tensor]:
    """Compute top-k predictions at each layer."""
    predictions = {}
    for layer_idx, hidden in hidden_states.items():
        normalized = norm(hidden)
        logits = lm_head(normalized)
        predictions[layer_idx] = logits.topk(k=10, dim=-1)
    return predictions
```

**Move `src/vlm_spectra/logit_lens/create_logit_lens.py` to `src/vlm_spectra/visualization/logit_lens_html.py`:**
- Keep HTML generation logic
- Import computation from `analysis/logit_lens.py`

**Run:** `uv run pytest`

---

### Phase 8: Cleanup and Final Tests

**Goal:** Clean up old files, update exports, run full test suite.

**Files to remove/deprecate:**
- `src/vlm_spectra/models/HookedVLM.py` - Replace with compatibility shim that imports from new location
- `src/vlm_spectra/models/ModelAdapter.py` - Split into base_adapter + qwen25_vl
- `src/vlm_spectra/models/model_prompts.py` - Move UI_TARS_PROMPT to qwen_processor or remove

**Create backward compat shim `src/vlm_spectra/models/HookedVLM.py`:**
```python
"""Backward compatibility - use vlm_spectra.HookedVLM instead."""
import warnings
from vlm_spectra.core.hooked_vlm import HookedVLM as NewHookedVLM

class HookedVLM(NewHookedVLM):
    def __init__(self, model_name="ByteDance-Seed/UI-TARS-1.5-7B", device="auto"):
        warnings.warn(
            "vlm_spectra.models.HookedVLM is deprecated. "
            "Use vlm_spectra.HookedVLM.from_pretrained() instead.",
            DeprecationWarning
        )
        # Create using new factory method
        instance = NewHookedVLM.from_pretrained(model_name, device=device)
        self.__dict__.update(instance.__dict__)
```

**Update `src/vlm_spectra/__init__.py` with clean exports:**
```python
from .core.hooked_vlm import HookedVLM
from .core.activation_cache import ActivationCache
from .models.registry import ModelRegistry
from .models.base_adapter import ModelAdapter
from .hooks import PatchResidualHook, PatchHeadHook, ZeroAblationHook

__all__ = [
    "HookedVLM",
    "ActivationCache",
    "ModelRegistry",
    "ModelAdapter",
    "PatchResidualHook",
    "PatchHeadHook",
    "ZeroAblationHook",
]
```

**Run full test suite:**
```bash
uv run pytest -v
uv run pytest --cov=src/vlm_spectra --cov-report=term-missing
```

**Verify backward compatibility:**
- All existing tests pass without modification
- Old import paths work (with deprecation warnings)

---

## Critical Files Reference

**Current files to decompose:**
| File | Lines | Destination |
|------|-------|-------------|
| `src/vlm_spectra/models/HookedVLM.py` | 768 | `core/hooked_vlm.py` (thin), `core/hook_manager.py`, `preprocessing/qwen_processor.py` |
| `src/vlm_spectra/models/ModelAdapter.py` | 354 | `models/base_adapter.py`, `models/adapters/qwen25_vl.py` |
| `src/vlm_spectra/utils/qwen_25_vl_utils.py` | 548 | `preprocessing/utils/vision_info.py` |
| `src/vlm_spectra/logit_lens/create_logit_lens.py` | 486 | `analysis/logit_lens.py`, `visualization/logit_lens_html.py` |
| `src/vlm_spectra/models/vlm_metadata.py` | 111 | `analysis/metadata.py` |
| `src/vlm_spectra/models/model_prompts.py` | 28 | Remove (make prompt a parameter) |

**Existing tests that must pass:**
| Test File | Tests | Validates |
|-----------|-------|-----------|
| `tests/test_run_with_cache.py` | 5 tests | Cache shapes, attention pattern accuracy |
| `tests/hookedvlm_test.py` | Core functionality | Model loading, generation, forward |
| `tests/test_batching.py` | Batch processing | Multiple inputs |
| `tests/test_prefill_functionality.py` | Assistant prefill | Preprocessing edge cases |

**Experiment patterns to extract (computer-interp/):**
| File | Pattern | Extract To |
|------|---------|------------|
| `square_patching_exp.py` | `ReplaceResidHook`, 7 patching modes | `hooks/patch_hooks.py` |
| `square_dla.py` | DLA computation | `analysis/direct_logit_attribution.py` (future) |
| `square_hidden_states.py` | Cache extraction patterns | Already in `HookManager` |

---

## Test Commands Summary

```bash
# Run all tests
uv run pytest

# Run unit tests only (fast, no GPU)
uv run pytest tests/unit/

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_run_with_cache.py

# Run with coverage
uv run pytest --cov=src/vlm_spectra --cov-report=term-missing

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```
