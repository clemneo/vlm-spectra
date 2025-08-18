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
        ["lm_attn_pattern"],
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
                 "lm_attn_pattern": (batch_size, model.adapter.lm_num_heads, seq_len, seq_len),
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

def test_attn_pattern_matches_output_attentions(model):
    """Test that lm_attn_pattern from run_with_cache matches output_attentions=True"""
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    
    # Get attention patterns using run_with_cache
    with model.run_with_cache(["lm_attn_pattern"]):
        _ = model.forward(inputs)
    
    # Get attention patterns using output_attentions=True
    outputs_attn = model.forward(inputs, output_attentions=True)
    
    # Verify that both methods return attention patterns
    assert hasattr(outputs_attn, 'attentions'), "output_attentions=True should return attentions"
    assert outputs_attn.attentions is not None, "attentions should not be None"
    assert len(outputs_attn.attentions) == model.adapter.lm_num_layers, f"Expected {model.adapter.lm_num_layers} attention layers"
    
    # Compare attention patterns layer by layer  
    tolerance = 2e-3  # Reasonable tolerance for bfloat16 precision across all layers
    
    for layer_idx in range(model.adapter.lm_num_layers):
        cache_key = ("lm_attn_pattern", layer_idx)
        assert cache_key in model.cache, f"Expected cache key {cache_key} not found"
        
        cache_attn = model.cache[cache_key]
        original_attn = outputs_attn.attentions[layer_idx]
        
        # Verify shapes match
        assert cache_attn.shape == original_attn.shape, f"Shape mismatch at layer {layer_idx}: cache={cache_attn.shape}, original={original_attn.shape}"
        
        # Verify attention patterns are valid (sum to 1 along last dimension)
        # Use higher tolerance for sum check due to bfloat16 precision
        sum_tolerance = 1e-2
        cache_sum = cache_attn.sum(dim=-1)
        original_sum = original_attn.sum(dim=-1)
        assert torch.allclose(cache_sum, torch.ones_like(cache_sum), atol=sum_tolerance), f"Cache attention sums invalid at layer {layer_idx}"
        assert torch.allclose(original_sum, torch.ones_like(original_sum), atol=sum_tolerance), f"Original attention sums invalid at layer {layer_idx}"
        
        # Compare the actual values (ensure tensors are on the same device)
        cache_attn_same_device = cache_attn.to(original_attn.device)
        diff = torch.abs(cache_attn_same_device - original_attn).mean()
        max_diff = torch.abs(cache_attn_same_device - original_attn).max()
        
        # Show the differences - this test will fail until RoPE is implemented
        print(f"Layer {layer_idx}: mean_abs_diff={diff:.6f}, max_abs_diff={max_diff:.6f}")
        
        assert diff < tolerance, f"Attention patterns differ at layer {layer_idx}: mean_abs_diff={diff:.6f} > tolerance={tolerance} (RoPE not implemented yet)"
    
    print(f"✓ All {model.adapter.lm_num_layers} attention pattern layers match between run_with_cache and output_attentions=True")

def test_resid_post_matches_output_hidden_states(model):
    """Test that lm_resid_post from run_with_cache matches output_hidden_states=True"""
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    
    # Get residual post activations using run_with_cache
    with model.run_with_cache(["lm_resid_post"]):
        _ = model.forward(inputs)
    
    # Get hidden states using output_hidden_states=True
    outputs_hidden = model.forward(inputs, output_hidden_states=True)
    
    # Verify that both methods return hidden states
    assert hasattr(outputs_hidden, 'hidden_states'), "output_hidden_states=True should return hidden_states"
    assert outputs_hidden.hidden_states is not None, "hidden_states should not be None"
    assert len(outputs_hidden.hidden_states) == model.adapter.lm_num_layers + 1, f"Expected {model.adapter.lm_num_layers + 1} hidden states (input + all layers)"
    
    # Compare residual post activations layer by layer  
    # Note: We exclude the last layer because models may apply additional normalization
    # to the final hidden state as mentioned in HuggingFace documentation
    tolerance = 2e-3  # Reasonable tolerance for bfloat16 precision across all layers
    
    layers_to_compare = model.adapter.lm_num_layers - 1  # Exclude the last layer
    
    for layer_idx in range(layers_to_compare):
        cache_key = ("lm_resid_post", layer_idx)
        assert cache_key in model.cache, f"Expected cache key {cache_key} not found"
        
        cache_resid = model.cache[cache_key]
        # Hidden states includes input embeddings at index 0, so layer outputs start at index 1
        original_hidden = outputs_hidden.hidden_states[layer_idx + 1]
        
        # Verify shapes match
        assert cache_resid.shape == original_hidden.shape, f"Shape mismatch at layer {layer_idx}: cache={cache_resid.shape}, original={original_hidden.shape}"
        
        # Compare the actual values (ensure tensors are on the same device)
        cache_resid_same_device = cache_resid.to(original_hidden.device)
        diff = torch.abs(cache_resid_same_device - original_hidden).mean()
        max_diff = torch.abs(cache_resid_same_device - original_hidden).max()
        
        # Show the differences
        print(f"Layer {layer_idx}: mean_abs_diff={diff:.6f}, max_abs_diff={max_diff:.6f}")
        
        assert diff < tolerance, f"Residual post activations differ at layer {layer_idx}: mean_abs_diff={diff:.6f} > tolerance={tolerance}"
    
    # Show the difference for the last layer (but don't assert on it)
    final_layer_idx = model.adapter.lm_num_layers - 1
    cache_key = ("lm_resid_post", final_layer_idx)
    if cache_key in model.cache:
        cache_resid = model.cache[cache_key]
        original_hidden = outputs_hidden.hidden_states[final_layer_idx + 1]
        cache_resid_same_device = cache_resid.to(original_hidden.device)
        diff = torch.abs(cache_resid_same_device - original_hidden).mean()
        max_diff = torch.abs(cache_resid_same_device - original_hidden).max()
        print(f"Layer {final_layer_idx} (final, not compared): mean_abs_diff={diff:.6f}, max_abs_diff={max_diff:.6f}")
    
    print(f"✓ {layers_to_compare} residual post layers match between run_with_cache and output_hidden_states=True (excluding final layer due to potential normalization)")

def test_final_layer_resid_post_vs_normalized_hidden_state(model):
    """Test the relationship between the final layer's lm_resid_post and the normalized hidden state"""
    image = generate_random_image()
    inputs = model.prepare_messages("Describe the image.", image)
    
    # Get residual post activations for the final layer using run_with_cache
    with model.run_with_cache(["lm_resid_post"]):
        _ = model.forward(inputs)
    
    # Get normalized hidden states using output_hidden_states=True
    outputs_hidden = model.forward(inputs, output_hidden_states=True)
    
    # Get the final layer's outputs
    final_layer_idx = model.adapter.lm_num_layers - 1
    cache_key = ("lm_resid_post", final_layer_idx)
    
    assert cache_key in model.cache, f"Expected cache key {cache_key} not found"
    
    # The cached resid_post is the output of the final transformer layer (before RMSNorm)
    final_resid_post = model.cache[cache_key]
    
    # The final hidden state from output_hidden_states includes RMSNorm normalization
    final_normalized_hidden = outputs_hidden.hidden_states[-1]  # Last hidden state
    
    # Verify shapes match
    assert final_resid_post.shape == final_normalized_hidden.shape, f"Shape mismatch: cache={final_resid_post.shape}, normalized={final_normalized_hidden.shape}"
    
    # Apply the same RMSNorm to our cached resid_post to see if it matches
    norm_layer = model.model.model.language_model.norm
    manually_normalized = norm_layer(final_resid_post.to(norm_layer.weight.device))
    
    # Compare manually normalized with HF's normalized output
    manually_normalized_same_device = manually_normalized.to(final_normalized_hidden.device)
    diff = torch.abs(manually_normalized_same_device - final_normalized_hidden).mean()
    max_diff = torch.abs(manually_normalized_same_device - final_normalized_hidden).max()
    
    print(f"Final layer before normalization vs after normalization:")
    print(f"  Raw resid_post vs normalized hidden state: mean_abs_diff={torch.abs(final_resid_post.to(final_normalized_hidden.device) - final_normalized_hidden).mean():.6f}")
    print(f"  Manually normalized vs HF normalized: mean_abs_diff={diff:.6f}, max_abs_diff={max_diff:.6f}")
    
    # The manually normalized version should match HF's normalized output very closely
    tolerance = 1e-4  # Tighter tolerance since we're applying the same normalization
    assert diff < tolerance, f"Manually normalized resid_post differs from HF normalized: mean_abs_diff={diff:.6f} > tolerance={tolerance}"
    
    print(f"✓ Final layer lm_resid_post + manual RMSNorm matches output_hidden_states final hidden state")