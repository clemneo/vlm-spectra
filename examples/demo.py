# %% [markdown]
# # Spectra: 1-Minute Live Demo
#
# **Three acts**: patch overview, activation caching, activation patching.
#
# Run **Cell 0** (install + model load) before presenting.
# Then execute **Cells 1-5** live.

# %%
# Cell 0 — Install + load model (run before presenting)
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

from vlm_spectra import HookedVLM


# %% [markdown]
# ---
# ## Act 1 — Patch Overview

# %%
# Cell 1 — Create image + patch overview
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
model = HookedVLM.from_pretrained(MODEL_NAME)
print(f"Loaded {MODEL_NAME.split('/')[-1]} — {model.lm_num_layers} layers, device={model.device}")

def make_circle(color, size=224):
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    m = size // 6
    draw.ellipse([m, m, size - m, size - m], fill=color)
    return img

red_img = make_circle("red")
blue_img = make_circle("blue")

info = model.process_image(red_img)
print(
    f"grid: {info.grid_h} x {info.grid_w} = {info.num_patches} patches, "
    f"effective patch size: {info.effective_patch_size}px"
)
model.generate_patch_overview(red_img, labels="every_10")

# %% [markdown]
# ## Act 2 — Activation Caching: P(color) across layers

# %%
# Cell 2 — Prepare inputs + cache activations
TASK = "What color is the circle?"
PREFILL = "The circle is"

red_inputs = model.prepare_messages(TASK, red_img, assistant_prefill=PREFILL)

tokenizer = model.processor.tokenizer
red_id = tokenizer.encode(" red", add_special_tokens=False)[0]
blue_id = tokenizer.encode(" blue", add_special_tokens=False)[0]

with model.run_with_cache(["lm.blocks.*.hook_resid_post"]) as cache:
    model.forward(red_inputs)

stacked = cache.stack("lm.blocks.*.hook_resid_post")  # [layers, batch, seq, hidden]

# %%
# Cell 3 — Project residuals through unembedding and plot
components = model.get_model_components()
norm, lm_head = components["norm"], components["lm_head"]

last_pos = stacked[:, 0, -1, :]  # [layers, hidden]
with torch.no_grad():
    probs = torch.softmax(lm_head(norm(last_pos)), dim=-1)

p_red = probs[:, red_id].float().cpu().numpy()
p_blue = probs[:, blue_id].float().cpu().numpy()

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(range(len(p_red)), p_red, color="red", linewidth=2, label='P("red")')
ax.plot(range(len(p_blue)), p_blue, color="blue", linewidth=2, label='P("blue")')
ax.set_xlabel("Layer")
ax.set_ylabel("Probability")
ax.set_title("Probability across layers — red circle image")
ax.legend()
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Act 3 — Activation Patching: image-only vs text-only

# %%
# Cell 4 — Cache blue activations + run patching loop across all layers
blue_inputs = model.prepare_messages(TASK, blue_img, assistant_prefill=PREFILL)
img_start, img_end = model.get_image_token_range(red_inputs)

# Cache blue (source) activations
with model.run_with_cache(["lm.blocks.*.hook_resid_post"]) as blue_cache:
    blue_output = model.forward(blue_inputs)
blue_activations = blue_cache.stack("lm.blocks.*.hook_resid_post").detach().clone()

# Compute normalization: logit_diff = logit("blue") - logit("red")
blue_logits = blue_output.logits[0, -1, :].detach()
blue_ld = (blue_logits[blue_id] - blue_logits[red_id]).item()

red_output = model.forward(red_inputs)
clean_logits = red_output.logits[0, -1, :].detach()
clean_ld = (clean_logits[blue_id] - clean_logits[red_id]).item()
denominator = blue_ld - clean_ld

# Patch at every layer: image-only and text-only
num_layers = model.lm_num_layers
patching_results = {"image_only": [], "text_only": []}

for layer_idx in tqdm(range(num_layers), desc="Patching layers"):
    hook_name = f"lm.blocks.{layer_idx}.hook_resid_post"
    blue_acts_layer = blue_activations[layer_idx]

    for condition in ["image_only", "text_only"]:
        if condition == "image_only":
            def patch_fn(module, args, kwargs, output,
                         _s=blue_acts_layer, _si=img_start, _ei=img_end):
                output[0, _si:_ei + 1, :] = _s[0, _si:_ei + 1, :]
                return output
        else:
            def patch_fn(module, args, kwargs, output,
                         _s=blue_acts_layer, _si=img_start, _ei=img_end):
                output[0, :_si, :] = _s[0, :_si, :]
                output[0, _ei + 1:, :] = _s[0, _ei + 1:, :]
                return output

        with model.run_with_hooks([(hook_name, patch_fn)]):
            patched_output = model.forward(red_inputs)

        patched_logits = patched_output.logits[0, -1, :]
        ld = (patched_logits[blue_id] - patched_logits[red_id]).item()
        norm_ld = (ld - clean_ld) / denominator if abs(denominator) > 1e-6 else 0.0
        patching_results[condition].append(norm_ld)

del blue_activations

# %%
# Cell 5 — Plot patching results
fig, ax = plt.subplots(figsize=(8, 4))
layers = list(range(num_layers))
ax.plot(layers, patching_results["image_only"], color="blue", linewidth=2, label="Image-only patching")
ax.plot(layers, patching_results["text_only"], color="orange", linewidth=2, label="Text-only patching")
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="No effect (clean red)")
ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="Full recovery (blue)")
ax.set_xlabel("Layer")
ax.set_ylabel("Normalized logit difference")
ax.set_title("Activation Patching: blue into red, by token type")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
