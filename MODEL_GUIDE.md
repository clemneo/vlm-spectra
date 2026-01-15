# Model guide

## Supported models

- ByteDance-Seed/UI-TARS-1.5-7B
- Qwen/Qwen2.5-VL-7B
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen3-VL-8B-Instruct

## Adding a new model

This guide describes the code changes needed to add a new VLM to `vlm-spectra`.

## 1) Implement a new adapter

Create a new adapter in `src/vlm_spectra/models/adapters/`. Use
`src/vlm_spectra/models/adapters/qwen25_vl.py` as a template and subclass
`ModelAdapter` from `src/vlm_spectra/models/base_adapter.py`.

Your adapter must provide:
- `MODEL_CLASS` and `PROCESSOR_CLASS` (Hugging Face classes used by `from_pretrained`).
- `SUPPORTED_MODELS` with the Hugging Face repo ids you want to accept.
- Implement all abstract properties/methods in `ModelAdapter`:
  - `lm_num_layers`, `lm_num_heads`, `lm_hidden_dim`, `lm_head_dim`
  - `get_lm_layer`, `get_lm_attn`, `get_lm_o_proj`, `get_lm_mlp`
  - `compute_per_head_contributions`
  - `compute_attention_patterns`
  - `get_image_token_id`
- Optionally implement `get_lm_norm` and `get_lm_head` for logit attribution.

## 2) Register the adapter

Register the adapter with the model registry so `HookedVLM.from_pretrained()` can
load it:

- Add the registry decorator in your adapter module:
  `@ModelRegistry.register("your_adapter_name")`
- Export it in `src/vlm_spectra/models/adapters/__init__.py`.
- Import the module in `src/vlm_spectra/models/registry.py` (see the `qwen25_vl`
  import at the bottom).

## 3) Update legacy adapter entry point (optional)

If you want `vlm_spectra.models.ModelAdapter.get_model_adapter()` to recognize the
new model class directly, update `src/vlm_spectra/models/ModelAdapter.py` with an
`isinstance` branch.

## 4) Update the web app default model (optional)

If the web demo should default to the new model, update
`src/vlm_spectra/web_app/model_manager.py` where `HookedVLM` is instantiated.

## 5) Add/extend tests

Update the parameter lists in:
- `tests/hookedvlm_test.py`
- `tests/test_batching.py`
- `tests/test_prefill_functionality.py`
- `tests/test_run_with_cache.py`

Add the new model id to the `pytest.mark.parametrize` lists and verify the
behavior matches existing expectations for `HookedVLM`. Be mindful of GPU memory
requirements for the larger tests.
