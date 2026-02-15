"""Patch grid visualization for processed images."""

from __future__ import annotations

from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from vlm_spectra.preprocessing.spatial import ImageInfo


def generate_patch_overview(
    info: ImageInfo,
    labels: Literal["none", "every_10", "all"] = "every_10",
    start_number: int = 0,
    highlight: list[int] | None = None,
    highlight_color: tuple[int, int, int, int] = (255, 200, 0, 100),
) -> Image.Image:
    """Draw a patch grid overlay on the processed image.

    Args:
        info: ImageInfo from processor.process_image().
        labels: Patch labelling mode — "none", "every_10", or "all".
        start_number: First patch number label.
        highlight: Patch indices to highlight.
        highlight_color: RGBA color for highlighting.

    Returns:
        Copy of the processed image with grid lines, optional labels, and
        optional patch highlighting.
    """
    eps = info.effective_patch_size
    resized_width, resized_height = info.processed_size

    overlay_image = info.image.copy()

    # Highlight patches (before grid lines so lines draw on top)
    if highlight is not None:
        r, g, b, alpha = highlight_color
        for idx in highlight:
            row, col = divmod(idx, info.grid_w)
            px1, py1 = col * eps, row * eps
            px2, py2 = px1 + eps, py1 + eps
            patch_overlay = Image.new("RGBA", (eps, eps), (r, g, b, alpha))
            overlay_image.paste(
                Image.alpha_composite(
                    overlay_image.crop((px1, py1, px2, py2)).convert("RGBA"),
                    patch_overlay,
                ).convert("RGB"),
                (px1, py1),
            )

    draw = ImageDraw.Draw(overlay_image)

    for i in range(info.grid_h):
        y_start = i * eps
        draw.line([(0, y_start), (resized_width, y_start)], fill="red", width=1)
        y_end = (i + 1) * eps - 1
        draw.line([(0, y_end), (resized_width, y_end)], fill="red", width=1)

    for j in range(info.grid_w):
        x_start = j * eps
        draw.line([(x_start, 0), (x_start, resized_height)], fill="red", width=1)
        x_end = (j + 1) * eps - 1
        draw.line([(x_end, 0), (x_end, resized_height)], fill="red", width=1)

    if labels != "none":
        font_size = max(14, min(32, eps // 3))

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except (OSError, IOError):
            try:
                font = ImageFont.load_default()
            except (OSError, IOError):
                font = None

        patch_num = start_number
        for i in range(info.grid_h):
            for j in range(info.grid_w):
                should_label = labels == "all" or (
                    patch_num % 10 == 0
                    or patch_num == start_number + info.num_patches - 1
                )
                if should_label:
                    x = j * eps + eps // 4
                    y = i * eps + eps // 4

                    text = str(patch_num)
                    if font:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    else:
                        text_width, text_height = len(text) * 6, 10

                    draw.rectangle(
                        [x - 1, y - 1, x + text_width + 1, y + text_height + 4],
                        fill="white",
                        outline="red",
                    )

                    draw.text((x, y), text, fill="red", font=font)

                patch_num += 1

    return overlay_image
