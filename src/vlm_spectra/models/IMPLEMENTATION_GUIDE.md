# Guide: Adding a New VLM to VLM Spectra

## Phase 1: Research the Model Architecture

Before writing any code, load the model in a Python REPL and answer these questions:

### 1. What is the HuggingFace model class?

- Find it in `transformers` (e.g., `LlavaForConditionalGeneration`)
- Check if it requires a specific transformers version

### 2. How do you access the language model?

- Print `model` and examine the structure
- Common patterns: `model.language_model`, `model.model.text_model`, `model.model`
- You need access to `.layers`, `.norm`, and the parent's `.lm_head`

### 3. What LLM architecture is it based on?

- Llama, Mistral, Qwen, etc.
- This determines which RoPE implementation to import
- Check if attention layers have `num_key_value_groups` (for GQA)

### 4. How is the image token identified?

- Check `model.config` for attributes like `image_token_index`, `image_token_id`
- Or check what token string the model uses (e.g., `<image>`, `<|image_pad|>`)

### 5. What's the image preprocessing strategy?

- Fixed size (e.g., 336x336) vs dynamic resolution
- This affects the processor implementation

---

## Phase 2: Create the Adapter

**File location:** `src/vlm_spectra/models/adapters/<model_name>.py`

### Key considerations:

1. **Guard the import** with try/except in case transformers doesn't have the model class - this allows the package to work even if the model isn't available

2. **Register with the decorator** `@ModelRegistry.register("your_model")` - the string is an internal identifier, not the HuggingFace name

3. **Set `SUPPORTED_MODELS`** to the full HuggingFace model names (e.g., `"llava-hf/llava-1.5-7b-hf"`)

4. **In `__init__`:**
   - Store a reference to the language model component
   - Call `set_attn_implementation("eager")` if available - needed for `output_attentions=True`

5. **Layer access properties** (`lm_layers`, `lm_attn`, `lm_o_proj`, `lm_mlp`):
   - Return the actual module lists from your language model
   - Attention is typically at `layer.self_attn`
   - MLP is typically at `layer.mlp`

6. **Projection getters** (`get_lm_q_proj`, `get_lm_k_proj`, `get_lm_v_proj`, `get_lm_gate_proj`, `get_lm_up_proj`, `get_lm_down_proj`):
   - Return the actual Linear modules
   - Names vary by architecture (e.g., `q_proj` vs `query`)

7. **Dimension properties** (`lm_num_layers`, `lm_num_heads`, `lm_hidden_dim`, `lm_head_dim`, `lm_num_kv_heads`, `lm_mlp_dim`):
   - Read from `self._language_model.config`
   - Attribute names vary (e.g., `num_hidden_layers`, `n_layers`)

8. **`compute_attention_scores`:**
   - Import the correct `apply_rotary_pos_emb` for your architecture
   - Import `repeat_kv` if using grouped-query attention
   - Check if your attention layer has `.scaling` or compute it as `head_dim ** -0.5`

9. **`get_image_token_id`:**
   - Return the token ID used for image patches
   - Either from config or via tokenizer lookup

10. **`format_cache_item`:**
    - Copy from an existing adapter (SmolVLM or LLaVA for Llama-based, Qwen for Qwen-based)
    - The logic handles reshaping tensors and unwrapping tuples

---

## Phase 3: Create the Processor

**File location:** `src/vlm_spectra/preprocessing/<model_name>_processor.py`

### Key considerations:

1. **Test the chat template first** in a REPL:
   ```python
   processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
   ```
   - Verify the output format looks correct
   - Check if the model uses `{"type": "image"}` or `{"type": "image", "image": pil_image}`

2. **Message format varies by model:**
   - Some put image before text, some after
   - Some require the actual PIL image in the message, others just a placeholder

3. **Generation prompt handling:**
   - Some models support `continue_final_message=True` for assistant prefill
   - Others need manual string concatenation after `add_generation_prompt=True`

4. **Batch processing:**
   - Ensure `padding=True` is passed for batched inputs
   - Some models need special handling for batched images

---

## Phase 4: Update HookedVLM

**File:** `src/vlm_spectra/core/hooked_vlm.py`

1. **Add import** for your processor class at the top

2. **Add condition** in `from_pretrained()` to select your processor:
   - Use a reliable substring match on `model_name`
   - Be specific enough to not accidentally match other models
   - Order matters - more specific checks should come before general ones

---

## Phase 5: Add Test Configuration

**File:** `tests/acceptance/conftest.py`

1. **Add to `MODEL_CAPABILITIES`:**
   - `contiguous_image_tokens`: Are all image tokens adjacent? (False for SmolVLM which interleaves)
   - `supports_batching`: Does batch processing work correctly?
   - `strict_residual_stream`: Does `hook_resid_post` equal the next layer's `hook_resid_pre`? (False for models with post-layer modifications)

2. **Add to `MODEL_ALIASES`:** Short convenient names for CLI usage

---

## Phase 6: Verify

1. **Check registration:**
   ```bash
   uv run python -c "from vlm_spectra.models.registry import ModelRegistry; print(ModelRegistry.list_supported_models())"
   ```

2. **Lint and format:**
   ```bash
   uv run ruff check src/vlm_spectra/models/adapters/<model_name>.py src/vlm_spectra/preprocessing/<model_name>_processor.py
   uv run ruff format src/vlm_spectra/models/adapters/<model_name>.py src/vlm_spectra/preprocessing/<model_name>_processor.py
   ```

3. **Run existing tests** (catch regressions):
   ```bash
   uv run pytest tests/
   ```

4. **Run acceptance tests against your model:**
   ```bash
   uv run pytest tests/acceptance/ --model <your-alias> -v
   ```

---

## Common Debugging Tips

- **"Module has no attribute 'layers'"**: Wrong path to language model - print and inspect the model structure
- **"No image tokens found"**: Wrong image token ID - check config attributes and tokenizer
- **Attention errors**: May need eager attention implementation or wrong RoPE import
- **Shape mismatches in cache**: Check `format_cache_item` handles your model's output format
- **Chat template errors**: Test `apply_chat_template` manually with your message format
