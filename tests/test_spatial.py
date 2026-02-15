import numpy as np
import pytest
from PIL import Image

from vlm_spectra.preprocessing.spatial import ImageInfo


def _make_info(orig_w=1024, orig_h=768, proc_w=1120, proc_h=784,
               crop_box=None, patch_size=14, spatial_merge_size=2):
    """Helper to create ImageInfo with a blank image."""
    img = Image.new("RGB", (proc_w, proc_h))
    return ImageInfo(
        image=img,
        original_size=(orig_w, orig_h),
        processed_size=(proc_w, proc_h),
        crop_box=crop_box,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        grid_h=proc_h // (patch_size * spatial_merge_size),
        grid_w=proc_w // (patch_size * spatial_merge_size),
    )


class TestImageInfoProperties:
    def test_effective_patch_size(self):
        info = _make_info()
        assert info.effective_patch_size == 28

    def test_grid_dimensions(self):
        info = _make_info()
        assert info.grid_h == 28   # 784 // 28
        assert info.grid_w == 40   # 1120 // 28

    def test_num_patches(self):
        info = _make_info()
        assert info.num_patches == 28 * 40


class TestCorrectBbox:
    def test_no_crop_simple_scale(self):
        """Bbox scaled from 1024x768 -> 1120x784."""
        info = _make_info()
        result = info.correct_bbox((0, 0, 512, 384))
        scale_x = 1120 / 1024
        scale_y = 784 / 768
        assert result.bbox == pytest.approx((0, 0, 512 * scale_x, 384 * scale_y))
        assert result.clipped is False

    def test_bbox_partially_outside(self):
        """Bbox that extends beyond image bounds gets clipped."""
        info = _make_info()
        result = info.correct_bbox((-10, -10, 512, 384))
        assert result.clipped is True
        assert result.bbox[0] == 0.0  # clamped to 0
        assert result.bbox[1] == 0.0

    def test_bbox_fully_outside(self):
        """Bbox entirely outside returns zero-area bbox."""
        info = _make_info()
        result = info.correct_bbox((-100, -100, -50, -50))
        assert result.clipped is True
        assert result.bbox == (0.0, 0.0, 0.0, 0.0)

    def test_with_crop(self):
        """Bbox corrected relative to crop region."""
        info = _make_info(crop_box=(100, 50, 924, 718))
        result = info.correct_bbox((200, 100, 400, 300))
        # Shifted: (100, 50, 300, 250), then scaled
        crop_w = 824
        assert result.bbox[0] == pytest.approx(100 * 1120 / crop_w)
        assert result.clipped is False


class TestCorrectSeg:
    def test_no_crop_resize(self):
        info = _make_info()
        mask = np.zeros((768, 1024), dtype=bool)
        mask[0:384, 0:512] = True  # top-left quadrant
        result = info.correct_seg(mask)
        assert result.mask.shape == (784, 1120)
        assert result.clipped is False
        # Top-left should be mostly True
        assert result.mask[0, 0] is np.bool_(True)
        # Bottom-right should be False
        assert result.mask[-1, -1] is np.bool_(False)

    def test_seg_outside_crop_is_clipped(self):
        info = _make_info(crop_box=(100, 100, 924, 668))
        mask = np.zeros((768, 1024), dtype=bool)
        mask[0:50, 0:50] = True  # entirely outside crop
        result = info.correct_seg(mask)
        assert result.clipped is True
        assert not result.mask.any()  # nothing survives


class TestCalcOverlay:
    def test_bbox_single_patch(self):
        """Bbox exactly covering first patch."""
        info = _make_info()
        eps = info.effective_patch_size  # 28
        result = info.calc_overlay(bbox=(0, 0, eps, eps))
        assert result.patch_indices == [0]
        assert result.overlap_fractions == [pytest.approx(1.0)]

    def test_bbox_spanning_patches(self):
        """Bbox covering 2x2 patches."""
        info = _make_info()
        eps = info.effective_patch_size
        result = info.calc_overlay(bbox=(0, 0, 2 * eps, 2 * eps))
        assert len(result.patch_indices) == 4
        assert all(f == pytest.approx(1.0) for f in result.overlap_fractions)
        # Patches: (0,0)=0, (0,1)=1, (1,0)=grid_w, (1,1)=grid_w+1
        assert set(result.patch_indices) == {0, 1, info.grid_w, info.grid_w + 1}

    def test_bbox_partial_overlap(self):
        """Bbox covering half of a patch."""
        info = _make_info()
        eps = info.effective_patch_size
        result = info.calc_overlay(bbox=(0, 0, eps / 2, eps))
        assert result.patch_indices == [0]
        assert result.overlap_fractions == [pytest.approx(0.5)]

    def test_mask_overlay(self):
        """Mask covering exactly one patch."""
        info = _make_info()
        eps = info.effective_patch_size
        mask = np.zeros((info.processed_size[1], info.processed_size[0]), dtype=bool)
        mask[0:eps, 0:eps] = True
        result = info.calc_overlay(mask=mask)
        assert result.patch_indices == [0]
        assert result.overlap_fractions == [pytest.approx(1.0)]

    def test_no_overlap(self):
        """Bbox outside all patches returns empty."""
        info = _make_info()
        result = info.calc_overlay(bbox=(-100, -100, -1, -1))
        assert result.patch_indices == []

    def test_must_provide_one(self):
        """Raises if neither bbox nor mask provided."""
        info = _make_info()
        with pytest.raises(ValueError):
            info.calc_overlay()


class TestLlavaProcessImage:
    """Tests for LlavaProcessor.process_image (no GPU needed)."""

    def _make_processor(self):
        from vlm_spectra.preprocessing.llava_processor import LlavaProcessor
        return LlavaProcessor(hf_processor=None)

    def test_square_image(self):
        """Square image: no cropping needed, crop_box is None."""
        proc = self._make_processor()
        img = Image.new("RGB", (500, 500), (128, 128, 128))
        info = proc.process_image(img)

        assert info.processed_size == (224, 224)
        assert info.original_size == (500, 500)
        assert info.grid_h == 16
        assert info.grid_w == 16
        assert info.num_patches == 256
        assert info.patch_size == 14
        assert info.spatial_merge_size == 1
        assert info.crop_box is None

    def test_landscape_image(self):
        """Landscape image: horizontal center crop."""
        proc = self._make_processor()
        img = Image.new("RGB", (800, 400), (128, 128, 128))
        info = proc.process_image(img)

        assert info.processed_size == (224, 224)
        assert info.grid_h == 16
        assert info.grid_w == 16
        # Landscape → crop_box should exist (horizontal crop)
        assert info.crop_box is not None
        x1, y1, x2, y2 = info.crop_box
        # y should span full height (no vertical crop)
        assert y1 == 0
        assert y2 == 400
        # x should be centered
        assert x1 > 0
        assert x2 < 800
        assert x2 - x1 < 800  # cropped horizontally

    def test_portrait_image(self):
        """Portrait image: vertical center crop."""
        proc = self._make_processor()
        img = Image.new("RGB", (400, 800), (128, 128, 128))
        info = proc.process_image(img)

        assert info.processed_size == (224, 224)
        assert info.grid_h == 16
        assert info.grid_w == 16
        # Portrait → crop_box should exist (vertical crop)
        assert info.crop_box is not None
        x1, y1, x2, y2 = info.crop_box
        # x should span full width (no horizontal crop)
        assert x1 == 0
        assert x2 == 400
        # y should be centered
        assert y1 > 0
        assert y2 < 800
        assert y2 - y1 < 800  # cropped vertically
