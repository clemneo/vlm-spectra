import torch
from vlm_spectra import HookedVLM
import pytest
from PIL import Image
import numpy as np

MODEL_NAMES = [
    "ByteDance-Seed/UI-TARS-1.5-7B",
    "Qwen/Qwen3-VL-8B-Instruct",
]

@pytest.fixture(scope="module", params=MODEL_NAMES)
def model(request):
    return HookedVLM.from_pretrained(request.param)

def generate_random_image(width=224, height=224, num_channels=3):
    random_array = np.random.randint(
        0, 255, (height, width, num_channels), dtype=np.uint8
    )
    random_image = Image.fromarray(random_array)
    return random_image

def test_prepare_messages_without_prefill(model):
    """Test that prepare_messages works as before when no prefill is provided"""
    image = generate_random_image()
    task = "Describe the image."
    
    # Test without prefill (current behavior)
    inputs, text = model.prepare_messages(task, image, return_text=True)
    
    # Should contain user message and generation prompt
    assert "user" in text.lower() or "human" in text.lower()
    assert inputs['input_ids'].shape[0] == 1  # batch size 1
    assert inputs['input_ids'].shape[1] > 0   # has tokens
    
    # Should end with assistant start token and newline (ready for generation)
    assert text.endswith("<|im_start|>assistant\n"), f"Expected to end with '<|im_start|>assistant\\n', got: '{text[-50:]}'"

def test_prepare_messages_with_prefill(model):
    """Test that prepare_messages correctly handles assistant prefill"""
    image = generate_random_image()
    task = "Describe the image."
    prefill = "This image contains"
    
    # Test with prefill
    inputs, text = model.prepare_messages(task, image, assistant_prefill=prefill, return_text=True)
    
    # Should contain both user message and assistant prefill
    assert "user" in text.lower() or "human" in text.lower()
    assert prefill in text
    
    # Should end with the prefill text (ready for continuation)
    assert text.endswith(f"<|im_start|>assistant\n{prefill}"), f"Expected to end with '<|im_start|>assistant\\n{prefill}', got: '{text[-100:]}'"
    
    assert inputs['input_ids'].shape[0] == 1  # batch size 1
    assert inputs['input_ids'].shape[1] > 0   # has tokens

def test_prepare_messages_with_empty_prefill(model):
    """Test that empty prefill behaves like no prefill"""
    image = generate_random_image()
    task = "Describe the image."
    
    # Test with empty prefill
    inputs_empty, text_empty = model.prepare_messages(task, image, assistant_prefill="", return_text=True)
    inputs_none, text_none = model.prepare_messages(task, image, return_text=True)
    
    # Both should be equivalent
    assert text_empty == text_none
    assert torch.equal(inputs_empty['input_ids'], inputs_none['input_ids'])

def test_generate_with_prefill(model):
    """Test that generate method works with assistant prefill"""
    image = generate_random_image()
    task = "Format your response as JSON."
    prefill = '{"description": "'
    
    # Generate with prefill
    inputs = model.prepare_messages(task, image, assistant_prefill=prefill)
    outputs = model.generate(inputs, max_new_tokens=50)
    
    # Decode the generated sequence
    generated_text = model.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # The prefill should appear in the generated text
    assert prefill in generated_text
    assert outputs.sequences.shape[0] == 1  # batch size 1
    assert outputs.sequences.shape[1] > inputs['input_ids'].shape[1]  # generated new tokens

def test_generate_with_json_prefill(model):
    """Test JSON prefill specifically as shown in documentation"""
    image = generate_random_image()
    task = "Can you format the answer in JSON with a 'description' field?"
    prefill = '{"description": "'
    
    inputs = model.prepare_messages(task, image, assistant_prefill=prefill)
    outputs = model.generate(inputs, max_new_tokens=100)
    
    generated_text = model.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Should contain the prefilled JSON structure
    assert '{"description": "' in generated_text
    # Should continue the JSON (look for quote closure and potential brace)
    assert generated_text.count('"') >= 2  # At least opening and some closing quotes

def test_forward_with_prefill(model):
    """Test that forward method works with assistant prefill"""
    image = generate_random_image()
    task = "Describe this image."
    prefill = "I can see"
    
    inputs = model.prepare_messages(task, image, assistant_prefill=prefill)
    outputs = model.forward(inputs)
    
    # Should return logits
    assert hasattr(outputs, 'logits')
    assert outputs.logits.shape[0] == 1  # batch size 1
    assert outputs.logits.shape[1] > 0   # sequence length > 0
    assert outputs.logits.shape[2] > 0   # vocab size > 0

def test_prefill_with_hooks(model):
    """Test that prefill works with the existing hook system"""
    image = generate_random_image()
    task = "Describe the image."
    prefill = "The image shows"
    
    inputs = model.prepare_messages(task, image, assistant_prefill=prefill)
    
    # Test with cache
    with model.run_with_cache(["lm_resid_post"]):
        model.forward(inputs)
    
    # Should have populated cache
    assert model.cache is not None
    assert len(model.cache) > 0
    
    # Check that we got expected cache structure
    cache_keys = list(model.cache.keys())
    assert any("lm_resid_post" in key[0] for key in cache_keys)

def test_prefill_comparison_with_no_prefill(model):
    """Test that prefill changes model behavior compared to no prefill"""
    image = generate_random_image()
    task = "What do you see?"
    prefill = "I observe a"
    
    # Generate without prefill
    inputs_no_prefill = model.prepare_messages(task, image)
    outputs_no_prefill = model.generate(inputs_no_prefill, max_new_tokens=30, do_sample=False)
    text_no_prefill = model.processor.tokenizer.decode(outputs_no_prefill.sequences[0], skip_special_tokens=True)
    
    # Generate with prefill
    inputs_with_prefill = model.prepare_messages(task, image, assistant_prefill=prefill)
    outputs_with_prefill = model.generate(inputs_with_prefill, max_new_tokens=30, do_sample=False)
    text_with_prefill = model.processor.tokenizer.decode(outputs_with_prefill.sequences[0], skip_special_tokens=True)
    
    # The prefilled version should contain the prefill text
    assert prefill in text_with_prefill
    # The texts should be different (prefill changes generation)
    assert text_no_prefill != text_with_prefill

def test_prefill_parameter_validation(model):
    """Test edge cases for prefill parameter"""
    image = generate_random_image()
    task = "Describe the image."
    
    # Test with None (should work like empty string)
    inputs_none = model.prepare_messages(task, image, assistant_prefill=None)
    inputs_empty = model.prepare_messages(task, image, assistant_prefill="")
    inputs_default = model.prepare_messages(task, image)
    
    # All should produce the same result
    assert torch.equal(inputs_none['input_ids'], inputs_empty['input_ids'])
    assert torch.equal(inputs_empty['input_ids'], inputs_default['input_ids'])
    
    # Test with whitespace-only prefill
    inputs_whitespace = model.prepare_messages(task, image, assistant_prefill="   ")
    # Should still work (whitespace is valid prefill)
    assert inputs_whitespace['input_ids'].shape[1] > 0

def test_long_prefill(model):
    """Test with longer prefill text"""
    image = generate_random_image()
    task = "Describe the image in detail."
    long_prefill = "This is a detailed analysis of the image. I can observe various elements including colors, shapes, and potential objects. The visual composition suggests"
    
    inputs = model.prepare_messages(task, image, assistant_prefill=long_prefill)
    outputs = model.generate(inputs, max_new_tokens=50)
    
    generated_text = model.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # The long prefill should appear in the generated text
    assert long_prefill in generated_text
    assert outputs.sequences.shape[1] > inputs['input_ids'].shape[1]  # generated additional tokens

def test_prefill_with_special_characters(model):
    """Test prefill with special characters and formatting"""
    image = generate_random_image()
    task = "Format as markdown."
    prefill = "# Analysis\n\n## Visual Elements:\n- "
    
    inputs = model.prepare_messages(task, image, assistant_prefill=prefill)
    outputs = model.generate(inputs, max_new_tokens=50)
    
    generated_text = model.processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # Should preserve the markdown formatting
    assert "# Analysis" in generated_text
    assert "## Visual Elements:" in generated_text
    assert "- " in generated_text
