from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class CorrectedBbox:
    """Bounding box corrected to processed-image coordinates."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in processed coords
    clipped: bool  # True if original bbox extended outside crop/image


@dataclass
class CorrectedSeg:
    """Segmentation mask corrected to processed-image coordinates."""

    mask: np.ndarray  # bool, shape (processed_h, processed_w)
    clipped: bool


@dataclass
class PatchOverlap:
    """Patches overlapping with a region of interest."""

    patch_indices: list[int]  # 0-based, row-major
    overlap_fractions: list[float]  # fraction of each patch covered (0.0-1.0)


class ImageInfo:
    """Processed image with spatial utilities. Created by processor.process_image()."""

    def __init__(
        self,
        image: Image.Image,
        original_size: tuple[int, int],
        processed_size: tuple[int, int],
        crop_box: tuple[int, int, int, int] | None,
        patch_size: int,
        spatial_merge_size: int,
        grid_h: int,
        grid_w: int,
    ) -> None:
        self.image = image
        self.original_size = original_size  # (w, h)
        self.processed_size = processed_size  # (w, h)
        self.crop_box = crop_box  # (x1, y1, x2, y2) | None
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.grid_h = grid_h
        self.grid_w = grid_w

    @property
    def effective_patch_size(self) -> int:
        return self.patch_size * self.spatial_merge_size

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    def correct_bbox(
        self, bbox: tuple[float, float, float, float]
    ) -> CorrectedBbox:
        """Correct a bbox from original image coords to processed image coords.

        Clamps to crop region (if any), shifts to crop-relative coords, then
        scales to processed dimensions.
        """
        x1, y1, x2, y2 = bbox
        orig_w, orig_h = self.original_size
        proc_w, proc_h = self.processed_size

        if self.crop_box is not None:
            cx1, cy1, cx2, cy2 = self.crop_box
        else:
            cx1, cy1, cx2, cy2 = 0, 0, orig_w, orig_h

        crop_w = cx2 - cx1
        crop_h = cy2 - cy1

        # Check if clipping occurs
        clipped = x1 < cx1 or y1 < cy1 or x2 > cx2 or y2 > cy2

        # Clamp to crop region
        x1_c = max(x1, cx1)
        y1_c = max(y1, cy1)
        x2_c = min(x2, cx2)
        y2_c = min(y2, cy2)

        # If fully outside, return zero-area bbox
        if x1_c >= x2_c or y1_c >= y2_c:
            return CorrectedBbox(bbox=(0.0, 0.0, 0.0, 0.0), clipped=True)

        # Shift to crop-relative coords
        x1_r = x1_c - cx1
        y1_r = y1_c - cy1
        x2_r = x2_c - cx1
        y2_r = y2_c - cy1

        # Scale to processed dimensions
        scale_x = proc_w / crop_w
        scale_y = proc_h / crop_h

        return CorrectedBbox(
            bbox=(x1_r * scale_x, y1_r * scale_y, x2_r * scale_x, y2_r * scale_y),
            clipped=clipped,
        )

    def correct_seg(self, mask: np.ndarray) -> CorrectedSeg:
        """Correct a segmentation mask from original image coords to processed image coords.

        Crops to crop region, resizes to processed dims via nearest-neighbor.
        """
        orig_w, orig_h = self.original_size
        proc_w, proc_h = self.processed_size

        if self.crop_box is not None:
            cx1, cy1, cx2, cy2 = self.crop_box
        else:
            cx1, cy1, cx2, cy2 = 0, 0, orig_w, orig_h

        # Crop the mask (mask is h x w)
        cropped = mask[cy1:cy2, cx1:cx2]

        # Check if any True pixels were lost
        clipped = bool(mask.any() and not cropped.any()) or bool(
            mask.sum() > cropped.sum()
        )

        # Resize via nearest-neighbor using PIL
        mask_img = Image.fromarray(cropped.astype(np.uint8) * 255)
        resized = mask_img.resize((proc_w, proc_h), Image.NEAREST)
        result_mask = np.array(resized) > 127

        return CorrectedSeg(mask=result_mask, clipped=clipped)

    def calc_overlay(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        mask: np.ndarray | None = None,
    ) -> PatchOverlap:
        """Find patches overlapping with a region. Inputs must be in processed-image coords.

        Provide exactly one of bbox or mask.
        """
        if bbox is None and mask is None:
            raise ValueError("Must provide either bbox or mask")
        if bbox is not None and mask is not None:
            raise ValueError("Must provide either bbox or mask, not both")

        eps = self.effective_patch_size
        proc_w, proc_h = self.processed_size
        patch_indices = []
        overlap_fractions = []

        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            # Clamp to image bounds
            bx1 = max(bx1, 0.0)
            by1 = max(by1, 0.0)
            bx2 = min(bx2, float(proc_w))
            by2 = min(by2, float(proc_h))

            if bx1 >= bx2 or by1 >= by2:
                return PatchOverlap(patch_indices=[], overlap_fractions=[])

            patch_area = eps * eps

            for row in range(self.grid_h):
                py1 = row * eps
                py2 = py1 + eps
                for col in range(self.grid_w):
                    px1 = col * eps
                    px2 = px1 + eps

                    # Intersection
                    ix1 = max(bx1, px1)
                    iy1 = max(by1, py1)
                    ix2 = min(bx2, px2)
                    iy2 = min(by2, py2)

                    if ix1 < ix2 and iy1 < iy2:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        frac = inter_area / patch_area
                        patch_indices.append(row * self.grid_w + col)
                        overlap_fractions.append(frac)

        else:
            assert mask is not None
            patch_area = eps * eps

            for row in range(self.grid_h):
                py1 = row * eps
                py2 = py1 + eps
                for col in range(self.grid_w):
                    px1 = col * eps
                    px2 = px1 + eps

                    patch_mask = mask[py1:py2, px1:px2]
                    count = int(patch_mask.sum())
                    if count > 0:
                        frac = count / patch_area
                        patch_indices.append(row * self.grid_w + col)
                        overlap_fractions.append(frac)

        return PatchOverlap(
            patch_indices=patch_indices, overlap_fractions=overlap_fractions
        )

    def patch_overview(
        self,
        labels: str = "every_10",
        start_number: int = 0,
        highlight: list[int] | None = None,
        highlight_color: tuple[int, int, int, int] = (255, 200, 0, 100),
    ) -> Image.Image:
        """Draw a patch grid overlay on the processed image.

        Delegates to :func:`vlm_spectra.visualization.patch_overview.generate_patch_overview`.
        """
        from vlm_spectra.visualization.patch_overview import (
            generate_patch_overview,
        )

        return generate_patch_overview(
            self,
            labels=labels,
            start_number=start_number,
            highlight=highlight,
            highlight_color=highlight_color,
        )
