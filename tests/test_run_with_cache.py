import torch
from vlm_spectra.models.HookedVLM import HookedVLM
import pytest
from PIL import Image
import numpy as np

MODEL_NAMES = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
]

@pytest.fixture(scope="session", params=MODEL_NAMES)
def model(request):
    return HookedVLM(request.param)

def generate_random_image(width=224, height=224, num_channels=3):
    random_array = np.random.randint(
        0, 255, (height, width, num_channels), dtype=np.uint8
    )
    random_image = Image.fromarray(random_array)
    return random_image

def test_run_with_cache(model):
    """Test the run_with_cache context manager functionality"""
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    seq_len = inputs['input_ids'].shape[1]
    
    # Test different cache sets
    cache_sets_to_test = [
        ["lm_resid_pre"],
        ["lm_attn_out"], 
        ["lm_mlp_out"],
        ["lm_resid_post"],
    ]
    
    for cache_set in cache_sets_to_test:
        # Test with forward pass
        with model.run_with_cache(cache_set):
            _ = model.forward(inputs)
        
        # Verify cache was populated
        assert model.cache is not None, f"Cache should not be None for cache_set: {cache_set}"
        assert isinstance(model.cache, dict), "Cache should be a dictionary"
        
        # Verify all requested cache keys are present
        for hook_pos in cache_set:
            # assert hook_pos in model.cache, f"Expected cache key '{hook_pos}' not found in cache"
            num_hook_pos = len([key for key in model.cache.keys() if hook_pos in key[0]])
            assert num_hook_pos == model.adapter.lm_num_layers, f"Expected {model.adapter.lm_num_layers} layers for {hook_pos}, got {num_hook_pos}"
            
            
        # Verify cache structure - should contain data for multiple layers
        for hook_pos in cache_set:
            # cache_data = model.cache[hook_pos]
            # assert len(cache_data) > 0, f"Cache for '{hook_pos}' should not be empty"
            # resid_pre shape: (batch_size, seq_len, hidden_size)
            # resid_out shape: (batch_size, seq_len, hidden_size)
            # resid_mid shape: (batch_size, seq_len, hidden_size)
            # resid_post shape: (batch_size, seq_len, hidden_size)
            # attn_out shape: (batch_size, seq_len, num_heads, head_dim)
            # mlp_out shape: (batch_size, seq_len, hidden_size)
            
            # print(model.cache[("resid_pre", 0)].shape)
            single_cache_item = model.cache[(hook_pos, 0)]
            assert isinstance(single_cache_item, torch.Tensor), f"Expected {hook_pos} to be a tensor, got {type(single_cache_item)}"
            
            batch_size = 1
            # check shapes
            correct_shapes = {
                "lm_resid_pre": (batch_size, seq_len, model.adapter.lm_hidden_dim),
                 "lm_resid_out": (batch_size, seq_len, model.adapter.lm_hidden_dim),
                 "lm_resid_mid": (batch_size, seq_len, model.adapter.lm_hidden_dim),
                 "lm_resid_post": (batch_size, seq_len, model.adapter.lm_hidden_dim),
                 "lm_attn_out": (batch_size, seq_len, model.adapter.lm_num_heads, model.adapter.lm_hidden_dim),
                 "lm_mlp_out": (batch_size, seq_len, model.adapter.lm_hidden_dim),
                }
            assert single_cache_item.shape == correct_shapes[hook_pos], f"Expected {hook_pos} to be a tensor of shape {correct_shapes[hook_pos]}, got {single_cache_item.shape}"

def test_run_with_cache_unsupported_hooks(model):
    """Test that unsupported hook positions raise appropriate errors"""
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    
    # Test unsupported hook positions
    with pytest.raises(NotImplementedError, match="Only LM hooks are supported"):
        with model.run_with_cache(["vision_resid_pre"]):
            model.forward(inputs)
            
    with pytest.raises(NotImplementedError, match="Resid_mid hooks are not supported"):
        with model.run_with_cache(["lm_resid_mid"]):
            model.forward(inputs)