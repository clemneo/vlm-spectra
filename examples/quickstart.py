# %% [markdown]
# # VLM Spectra Quickstart
#
# This notebook demonstrates the core functionality of VLM Spectra, a toolkit
# for Vision-Language Model (VLM) interpretability. You'll learn how to:
#
# 1. Load a VLM with hooks for interpretability
# 2. Generate text from images
# 3. Run forward passes and inspect logits
# 4. Cache activations from intermediate layers
# 5. Access model components (norm, lm_head, tokenizer)
# 6. Create interactive logit lens visualizations
# 7. Visualize image patches
# 8. Perform basic activation patching
#
# **Note**: By default this notebook uses Qwen3-VL-8B-Instruct. For faster
# testing on limited hardware, you can switch to `HuggingFaceTB/SmolVLM-256M-Instruct`.

# %%
# Imports and Setup
import torch
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

from vlm_spectra import HookedVLM, PatchActivation, ActivationCache

# %% [markdown]
# ## Cell 2: Load Model
#
# We use `HookedVLM.from_pretrained()` to load a vision-language model with
# interpretability hooks. The model is automatically placed on GPU if available.

# %%
# Load the model (change to "HuggingFaceTB/SmolVLM-256M-Instruct" for faster testing)
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"Loading model: {MODEL_NAME}")
model = HookedVLM.from_pretrained(MODEL_NAME)

print(f"Model loaded successfully!")
print(f"  - Number of layers: {model.lm_num_layers}")
print(f"  - Device: {model.device}")

# %% [markdown]
# ## Cell 3: Create/Load Image
#
# We'll create a simple test image. You can also load your own image using
# `Image.open("path/to/image.png")`.

# %%
def create_test_image(size: int = 224, color: tuple = (100, 150, 200)) -> Image.Image:
    """Create a simple test image with a colored square."""
    image = Image.new("RGB", (size, size), (240, 240, 240))  # Light gray background
    draw = ImageDraw.Draw(image)
    # Draw a colored rectangle in the center
    margin = size // 4
    draw.rectangle([margin, margin, size - margin, size - margin], fill=color)
    return image

# Create a test image
image = create_test_image(224, color=(50, 100, 200))  # Blue square
print(f"Created test image: {image.size}")

# Prepare inputs using model.prepare_messages()
prompt = "Describe this image in one sentence."
inputs = model.prepare_messages(prompt, image)

print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")

# %% [markdown]
# ## Cell 4: Basic Generation
#
# Use `model.generate()` to generate text from the image. This is the standard
# way to get model outputs.

# %%
# Generate text
outputs = model.generate(inputs, max_new_tokens=50, do_sample=False)

# Decode the output
generated_ids = outputs.sequences[0]
generated_text = model.processor.tokenizer.decode(
    generated_ids, skip_special_tokens=True
)
print("Generated text:")
print(generated_text)

# %% [markdown]
# ## Cell 5: Forward Pass
#
# Use `model.forward()` to get the raw logits without generating. This is useful
# for analyzing model predictions at specific positions.

# %%
# Run forward pass
outputs = model.forward(inputs)

# Inspect logits shape: [batch, seq_len, vocab_size]
print(f"Logits shape: {outputs.logits.shape}")

# Get top predictions for the last token position
last_logits = outputs.logits[0, -1, :]
probs = torch.softmax(last_logits, dim=-1)
top_probs, top_ids = probs.topk(5)

print("\nTop 5 next token predictions:")
for prob, tok_id in zip(top_probs, top_ids):
    token = model.processor.tokenizer.decode(tok_id.item())
    print(f"  {token!r}: {prob.item():.4f}")

# %% [markdown]
# ## Cell 6: Activation Caching
#
# Use `model.run_with_cache()` to capture activations from intermediate layers.
# This is essential for interpretability analysis.

# %%
# Capture residual stream activations at all layers
with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
    model.forward(inputs)

# Access the cache (stored in model.cache as a dict)
cache = model.cache
print(f"Cached {len(cache)} activation tensors")

# Access a specific layer's activation
layer_5_activation = cache["lm.blocks.5.hook_resid_post"]
print(f"\nLayer 5 activation shape: {layer_5_activation.shape}")
print(f"  [batch, seq_len, hidden_dim]")

# Stack all layers into a single tensor using ActivationCache
# First, wrap the dict in an ActivationCache to use stack()
activation_cache = ActivationCache()
activation_cache._data = cache

stacked = activation_cache.stack("lm.blocks.*.hook_resid_post")
print(f"\nStacked activations shape: {stacked.shape}")
print(f"  [num_layers, batch, seq_len, hidden_dim]")

# %% [markdown]
# ## Cell 7: Model Components
#
# Access internal model components needed for logit lens and other analyses.

# %%
# Get model components
components = model.get_model_components()

print("Available components:")
print(f"  - norm: {type(components['norm']).__name__}")
print(f"  - lm_head: {type(components['lm_head']).__name__}")
print(f"  - tokenizer: {type(components['tokenizer']).__name__}")

# Get image token range - useful for focusing analysis on image tokens
image_start, image_end = model.get_image_token_range(inputs)
num_image_tokens = image_end - image_start + 1
print(f"\nImage tokens: positions {image_start} to {image_end} ({num_image_tokens} tokens)")

# %% [markdown]
# ## Cell 8: Logit Lens Visualization
#
# Create an interactive HTML visualization showing what the model "thinks" at
# each layer for each token position. This requires metadata extraction.

# %%
from vlm_spectra.visualization.logit_lens_html import create_logit_lens
from vlm_spectra.analysis.metadata import VLMMetadataExtractor

# Make sure we have cached activations
with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
    model.forward(inputs)

# Wrap cache for stacking
activation_cache = ActivationCache()
activation_cache._data = model.cache
hidden_states = activation_cache.stack("lm.blocks.*.hook_resid_post")

# Get model components
components = model.get_model_components()
norm = components["norm"]
lm_head = components["lm_head"]
tokenizer = components["tokenizer"]

# Extract metadata for visualization
metadata = VLMMetadataExtractor.extract_metadata_qwen(
    model=model.model,
    processor=model.processor,
    inputs=inputs,
    original_image=image,
)

print("Extracted metadata:")
print(f"  - Token labels: {len(metadata['token_labels'])} tokens")
print(f"  - Image size: {metadata['image_size']}")
print(f"  - Grid size: {metadata['grid_size']} (patches)")
print(f"  - Patch size: {metadata['patch_size']}px")

# Create output directory
output_dir = Path("./tmp")
output_dir.mkdir(exist_ok=True)

# Create the logit lens visualization
create_logit_lens(
    hidden_states=hidden_states,
    norm=norm,
    lm_head=lm_head,
    tokenizer=tokenizer,
    image=image,
    token_labels=metadata["token_labels"],
    image_size=metadata["image_size"],
    grid_size=metadata["grid_size"],
    patch_size=metadata["patch_size"],
    model_name=MODEL_NAME.split("/")[-1],
    image_filename="test_image.png",
    prompt=prompt,
    save_folder=str(output_dir),
)

print(f"\nLogit lens HTML saved to {output_dir}/")

# %% [markdown]
# ## Cell 9: Patch Overview Visualization
#
# Visualize how the image is divided into patches. Each patch corresponds to
# image tokens in the model's input.

# %%
# Generate patch overview
patch_overview = model.generate_patch_overview(image, labels="every_10")

# Save or display the patch overview
patch_overview.save(output_dir / "patch_overview.png")
print(f"Patch overview saved to {output_dir / 'patch_overview.png'}")
print(f"Patch overview size: {patch_overview.size}")

# %% [markdown]
# ## Cell 10: Activation Patching Basics
#
# Activation patching lets you perform causal interventions - replace activations
# from one input with activations from another to understand what information
# is encoded where.

# %%
# Create two test images with different colors
red_image = create_test_image(224, color=(200, 50, 50))   # Red square
blue_image = create_test_image(224, color=(50, 50, 200))  # Blue square

# Prepare inputs for both
red_inputs = model.prepare_messages("What color is the square?", red_image)
blue_inputs = model.prepare_messages("What color is the square?", blue_image)

# Capture activations from the blue image (corrupted run)
with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
    model.forward(blue_inputs)
blue_cache = {k: v.clone() for k, v in model.cache.items()}

# Get baseline prediction on red image
outputs_clean = model.forward(red_inputs)
clean_logits = outputs_clean.logits[0, -1, :]

# Get token IDs for "red" and "blue"
red_token_id = model.processor.tokenizer.encode(" red", add_special_tokens=False)[0]
blue_token_id = model.processor.tokenizer.encode(" blue", add_special_tokens=False)[0]

print("Baseline (red image):")
print(f"  P(red): {torch.softmax(clean_logits, dim=-1)[red_token_id].item():.4f}")
print(f"  P(blue): {torch.softmax(clean_logits, dim=-1)[blue_token_id].item():.4f}")

# Now patch: replace layer 10's activation at position 50 with blue image's activation
layer_to_patch = min(10, model.lm_num_layers - 1)
position_to_patch = 50  # Adjust based on your sequence length

# Get the replacement activation from blue cache
replacement = blue_cache[f"lm.blocks.{layer_to_patch}.hook_resid_post"][0, position_to_patch, :].clone()

# Create the patch hook
patch_hook = PatchActivation(
    replacement=replacement,
    token_idx=position_to_patch,
)

# Run with the patch
with model.run_with_hooks([(f"lm.blocks.{layer_to_patch}.hook_resid_post", patch_hook)]):
    outputs_patched = model.forward(red_inputs)

patched_logits = outputs_patched.logits[0, -1, :]

print(f"\nAfter patching layer {layer_to_patch}, position {position_to_patch}:")
print(f"  P(red): {torch.softmax(patched_logits, dim=-1)[red_token_id].item():.4f}")
print(f"  P(blue): {torch.softmax(patched_logits, dim=-1)[blue_token_id].item():.4f}")

# Calculate the change
red_prob_change = (
    torch.softmax(patched_logits, dim=-1)[red_token_id].item() -
    torch.softmax(clean_logits, dim=-1)[red_token_id].item()
)
print(f"\nChange in P(red): {red_prob_change:+.4f}")

# %% [markdown]
# ## Summary
#
# This quickstart covered the core VLM Spectra functionality:
#
# - **HookedVLM**: The main wrapper for loading VLMs with interpretability hooks
# - **generate()** / **forward()**: Standard inference methods
# - **run_with_cache()**: Capture intermediate activations
# - **run_with_hooks()**: Patch activations for causal interventions
# - **get_model_components()**: Access norm, lm_head, tokenizer
# - **get_image_token_range()**: Find image token positions
# - **generate_patch_overview()**: Visualize image patches
# - **create_logit_lens()**: Interactive HTML visualization
#
# For more advanced usage, see:
# - `examples/activation_patching_heatmap.py` - Full activation patching experiment
# - `tests/acceptance/test_core_contracts.py` - API reference tests
