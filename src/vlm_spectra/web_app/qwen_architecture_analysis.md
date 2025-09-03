# Qwen2.5-VL Architecture Analysis: Why Small Images Fail

## Key Findings

Based on examination of the Hugging Face transformers source code for Qwen2.5-VL, here's why small images perform poorly:

## 1. **Fixed Window Size Architecture**

From `configuration_qwen2_5_vl.py`:
- `window_size = 112` (fixed parameter)
- `patch_size = 14` 
- `spatial_merge_size = 2`

## 2. **Critical Window Size Calculation**

From `modeling_qwen2_5_vl.py` line 442:
```python
vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
# vit_merger_window_size = 112 // 2 // 14 = 4
```

This means the model expects **at least 4x4 patches** per attention window.

## 3. **Window Padding Logic**

Lines 450-451:
```python
pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
```

**For small images:**
- 84x84 image → 6x6 patches → 3x3 after spatial merge → **NEEDS PADDING**
- Padding fills with `-100` (invalid tokens)
- Model sees mostly padding tokens, not real image data

**For large images:**
- 476x476 image → 34x34 patches → 17x17 after spatial merge → **FITS NATURALLY**
- 17x17 = 4×4 + 1×1, so minimal padding needed
- Model sees mostly real image data

## 4. **Spatial Merge Process**

The model uses `spatial_merge_size = 2`, meaning:
- Every 2x2 patch group gets merged into 1 token
- This reduces the effective resolution by half
- Small images become even smaller after merging

## 5. **Attention Window Processing**

Lines 455-467: The model processes images in fixed 4x4 windows after spatial merging.

**Small image processing:**
1. 84x84 → 6x6 patches → 3x3 after merge
2. Need 4x4 window → pad with `-100` tokens
3. Window contains: ~56% real data, 44% padding
4. Attention patterns get corrupted by padding

**Large image processing:**
1. 476x476 → 34x34 patches → 17x17 after merge  
2. 17x17 fits into 4×4 windows efficiently
3. Windows contain: ~85% real data, 15% padding
4. Attention patterns remain coherent

## 6. **Why Coordinate Prediction Fails**

The model's coordinate prediction relies on:
- **Spatial attention patterns** between patches
- **Positional relationships** within attention windows  
- **Coherent feature representations** across the image

With small images:
- Attention windows are mostly padding
- Spatial relationships are obscured  
- Feature representations become noisy
- Coordinate predictions become random

## 7. **Optimal Image Sizes**

The model works best when:
- Image dimensions are multiples of `window_size = 112`
- After patch division (÷14) and spatial merge (÷2), the result fits cleanly into 4x4 windows
- Optimal sizes: 224x224, 336x336, 448x448, 560x560, etc.

## Conclusion

**Qwen2.5-VL is architecturally designed for medium-to-large images** due to its fixed windowed attention mechanism. Small images don't provide enough spatial context for the attention windows to function properly, leading to degraded performance on spatial reasoning tasks like coordinate prediction.

This explains why our demo needed 476x476 images (17×17 patches → fits 4×4 windows well) rather than 84x84 images (6×6 patches → mostly padding in 4×4 windows).