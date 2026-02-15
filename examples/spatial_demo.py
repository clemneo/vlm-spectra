# %% [markdown]
# # Spatial Preprocessing Demo
#
# This script demonstrates the spatial preprocessing utilities in VLM Spectra:
#
# - **`ImageInfo`**: Metadata about how an image was resized/cropped for the model
# - **`correct_bbox()`**: Map bounding boxes from original to processed coordinates
# - **`correct_seg()`**: Map segmentation masks to processed coordinates
# - **`calc_overlay()`**: Find which vision patches overlap a region of interest
#
# **No GPU or model required** — we use `QwenProcessor.process_image()` directly.

# %%
# Imports
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

from vlm_spectra.preprocessing.qwen_processor import QwenProcessor
from vlm_spectra.preprocessing.spatial import ImageInfo
from vlm_spectra.visualization import generate_patch_overview

# %%
# Create a synthetic dashboard screenshot with two elements of interest:
#   1. A rectangular "Save Changes" button (for bbox demo)
#   2. A circular user avatar (for segmentation mask demo)
SCREEN_W, SCREEN_H = 1920, 1080

screen = Image.new("RGB", (SCREEN_W, SCREEN_H), (245, 245, 248))
draw = ImageDraw.Draw(screen)

# -- Top nav bar --
draw.rectangle([0, 0, SCREEN_W, 56], fill=(25, 30, 40))
draw.text((24, 18), "Dashboard", fill=(255, 255, 255))
draw.text((SCREEN_W - 200, 18), "Settings  |  Logout", fill=(180, 180, 190))

# -- Sidebar --
draw.rectangle([0, 56, 260, SCREEN_H], fill=(35, 40, 52))
for i, label in enumerate(["Overview", "Analytics", "Users", "Reports", "Settings"]):
    y = 80 + i * 44
    fill = (55, 60, 75) if i == 2 else (35, 40, 52)
    draw.rectangle([0, y, 260, y + 44], fill=fill)
    draw.text((28, y + 12), label, fill=(200, 200, 210))

# -- Main content: User profile card --
card_x, card_y = 320, 100
card_w, card_h = 800, 500
draw.rounded_rectangle(
    [card_x, card_y, card_x + card_w, card_y + card_h],
    radius=12, fill=(255, 255, 255), outline=(220, 220, 225),
)
draw.text((card_x + 28, card_y + 20), "User Profile", fill=(30, 30, 30))
draw.line(
    [(card_x + 20, card_y + 52), (card_x + card_w - 20, card_y + 52)],
    fill=(230, 230, 235), width=1,
)

# -- Circular avatar (non-rectangular element for segmask) --
AVATAR_CENTER = (card_x + 120, card_y + 160)
AVATAR_RADIUS = 56
ax, ay = AVATAR_CENTER
draw.ellipse(
    [ax - AVATAR_RADIUS, ay - AVATAR_RADIUS, ax + AVATAR_RADIUS, ay + AVATAR_RADIUS],
    fill=(70, 130, 210),
)
draw.text((ax - 12, ay - 8), "JD", fill=(255, 255, 255))

# -- Profile details --
draw.text((card_x + 210, card_y + 100), "Jane Doe", fill=(30, 30, 30))
draw.text((card_x + 210, card_y + 130), "jane.doe@example.com", fill=(120, 120, 130))
draw.text((card_x + 210, card_y + 160), "Role: Administrator", fill=(120, 120, 130))
draw.text((card_x + 210, card_y + 190), "Last login: 2 hours ago", fill=(120, 120, 130))

# Form fields
for i, label in enumerate(["Display Name", "Email", "Department"]):
    fy = card_y + 260 + i * 64
    draw.text((card_x + 28, fy), label, fill=(80, 80, 90))
    draw.rounded_rectangle(
        [card_x + 28, fy + 20, card_x + card_w - 28, fy + 50],
        radius=4, fill=(250, 250, 252), outline=(200, 200, 210),
    )

# -- "Save Changes" button (rectangular element for bbox) --
BUTTON_BBOX = (card_x + card_w - 200, card_y + card_h - 60, card_x + card_w - 28, card_y + card_h - 20)
bx1, by1, bx2, by2 = BUTTON_BBOX
draw.rounded_rectangle([bx1, by1, bx2, by2], radius=6, fill=(50, 120, 220))
draw.text((bx1 + 28, by1 + 8), "Save Changes", fill=(255, 255, 255))

# -- Activity panel on the right --
panel_x = card_x + card_w + 40
draw.rounded_rectangle(
    [panel_x, 100, panel_x + 440, 460],
    radius=12, fill=(255, 255, 255), outline=(220, 220, 225),
)
draw.text((panel_x + 20, 120), "Recent Activity", fill=(30, 30, 30))
for i, activity in enumerate([
    "Updated profile photo",
    "Changed password",
    "Added new team member",
    "Exported monthly report",
    "Modified permissions",
]):
    draw.text((panel_x + 20, 160 + i * 36), f"  {activity}", fill=(100, 100, 110))

print(f"Synthetic screenshot: {screen.size}")

# %%
# Process the image through QwenProcessor (no HF processor needed)
proc = QwenProcessor(hf_processor=None)
info: ImageInfo = proc.process_image(screen)

print(f"Original size:   {info.original_size}")
print(f"Processed size:  {info.processed_size}")
print(f"Grid (h x w):    {info.grid_h} x {info.grid_w}")
print(f"Effective patch:  {info.effective_patch_size}px")
print(f"Num patches:     {info.num_patches}")

# %% [markdown]
# ## Bounding Box: "Save Changes" Button
#
# Use `correct_bbox` to map the button's coordinates from the original screenshot
# to the processed image, then find which patches overlap it.

# %%
# Correct the button bbox to processed coords
corrected_btn = info.correct_bbox(BUTTON_BBOX)

print(f"Original bbox:   {BUTTON_BBOX}")
print(f"Corrected bbox:  {tuple(round(v, 1) for v in corrected_btn.bbox)}")
print(f"Clipped:         {corrected_btn.clipped}")

# Find overlapping patches
overlap_btn = info.calc_overlay(bbox=corrected_btn.bbox)
print(f"Overlapping patches: {overlap_btn.patch_indices}")
print(f"Overlap fractions:   {[round(f, 3) for f in overlap_btn.overlap_fractions]}")

# %% [markdown]
# ## Segmentation Mask: Circular Avatar
#
# The avatar is circular — a bounding box would include corners that aren't part
# of the avatar. A segmentation mask captures the exact shape, so `correct_seg`
# and `calc_overlay(mask=...)` give more precise patch coverage.

# %%
# Create a circular mask for the avatar in original image coords
avatar_mask = np.zeros((SCREEN_H, SCREEN_W), dtype=bool)
yy, xx = np.ogrid[:SCREEN_H, :SCREEN_W]
dist = (xx - AVATAR_CENTER[0]) ** 2 + (yy - AVATAR_CENTER[1]) ** 2
avatar_mask[dist <= AVATAR_RADIUS**2] = True

print(f"Avatar mask pixels: {avatar_mask.sum()}")

# Correct the mask to processed image coords
corrected_avatar = info.correct_seg(avatar_mask)
print(f"Corrected mask shape: {corrected_avatar.mask.shape}")
print(f"Clipped:              {corrected_avatar.clipped}")
print(f"True pixels:          {corrected_avatar.mask.sum()}")

# Find overlapping patches
overlap_avatar = info.calc_overlay(mask=corrected_avatar.mask)
print(f"Overlapping patches: {overlap_avatar.patch_indices}")
print(f"Overlap fractions:   {[round(f, 3) for f in overlap_avatar.overlap_fractions]}")

# Compare with a bbox approach for the same avatar
avatar_bbox = (
    AVATAR_CENTER[0] - AVATAR_RADIUS,
    AVATAR_CENTER[1] - AVATAR_RADIUS,
    AVATAR_CENTER[0] + AVATAR_RADIUS,
    AVATAR_CENTER[1] + AVATAR_RADIUS,
)
corrected_avatar_bbox = info.correct_bbox(avatar_bbox)
overlap_avatar_bbox = info.calc_overlay(bbox=corrected_avatar_bbox.bbox)
print(f"\nBbox approach finds {len(overlap_avatar_bbox.patch_indices)} patches")
print(f"Mask approach finds  {len(overlap_avatar.patch_indices)} patches")
print("(Mask is more precise for non-rectangular regions)")

# %% [markdown]
# ## Visualization
#
# `generate_patch_overview` draws the patch grid and can highlight specific patches.
# The `labels` parameter controls labelling: `"none"`, `"every_10"`, or `"all"`.

# %%
output_dir = Path("./tmp")
output_dir.mkdir(exist_ok=True)

all_patches = list(set(overlap_btn.patch_indices + overlap_avatar.patch_indices))
cx1, cy1, cx2, cy2 = corrected_btn.bbox

# 1. Base processed image (no grid)
info.image.save(output_dir / "1_base.png")
print(f"Saved {output_dir / '1_base.png'}")

# 2. Base image with bbox and segmask regions drawn on
vis_regions = info.image.copy()
d = ImageDraw.Draw(vis_regions)
d.rectangle([cx1, cy1, cx2, cy2], outline=(0, 255, 0), width=2)
# Draw the corrected avatar mask contour by finding edge pixels
mask = corrected_avatar.mask.astype(np.uint8) * 255
mask_img = Image.fromarray(mask).filter(ImageFilter.FIND_EDGES)
vis_regions_rgba = vis_regions.convert("RGBA")
contour_overlay = Image.new("RGBA", vis_regions.size, (0, 0, 0, 0))
contour_draw = ImageDraw.Draw(contour_overlay)
edge_arr = np.array(mask_img)
ys, xs = np.where(edge_arr > 127)
for x, y in zip(xs, ys):
    contour_draw.point((x, y), fill=(0, 180, 255, 255))
vis_regions = Image.alpha_composite(vis_regions_rgba, contour_overlay).convert("RGB")
vis_regions.save(output_dir / "2_regions.png")
print(f"Saved {output_dir / '2_regions.png'}")

# 3. Grid with highlighted patches (no labels)
vis_highlight = generate_patch_overview(
    info, labels="none", highlight=all_patches,
)
ImageDraw.Draw(vis_highlight).rectangle([cx1, cy1, cx2, cy2], outline=(0, 255, 0), width=2)
vis_highlight.save(output_dir / "3_grid_highlight.png")
print(f"Saved {output_dir / '3_grid_highlight.png'}")

# 4. Grid with highlighted patches and every-10th label
vis_labels = generate_patch_overview(
    info, labels="every_10", highlight=all_patches,
)
ImageDraw.Draw(vis_labels).rectangle([cx1, cy1, cx2, cy2], outline=(0, 255, 0), width=2)
vis_labels.save(output_dir / "4_grid_highlight_labels.png")
print(f"Saved {output_dir / '4_grid_highlight_labels.png'}")

# 5. Grid with highlighted patches and all labels
vis_all_labels = generate_patch_overview(
    info, labels="all", highlight=all_patches,
)
ImageDraw.Draw(vis_all_labels).rectangle([cx1, cy1, cx2, cy2], outline=(0, 255, 0), width=2)
vis_all_labels.save(output_dir / "5_grid_highlight_all_labels.png")
print(f"Saved {output_dir / '5_grid_highlight_all_labels.png'}")
